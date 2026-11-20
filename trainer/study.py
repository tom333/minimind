"""
Étude Optuna : une étude = une campagne d'entraînements (trials) décrite par un fichier YAML.

Chaque trial lance train_pretrain.py dans un sous-processus (GPU et compilation repartent de zéro),
lit sa sortie en direct pour suivre la val_loss, et peut l'arrêter s'il est mal parti (pruner).
L'étude est stockée dans une base SQLite : relancer la même étude la REPREND où elle s'était arrêtée.

Fichier d'étude (cf. trainer/etudes/*.yaml) :
  name: muon-lr                 # nom de l'étude (et de l'expérience MLflow : minimind-<name>)
  sampler: grid                 # grid = toutes les combinaisons ; tpe = recherche guidée ; random
  n_trials: 12                  # ignoré pour grid
  pruner: none                  # none | median (arrête un trial sous la médiane des autres)
  base: {max_tokens: 50000000}  # arguments communs passés à train_pretrain.py
  data_file: sft_t2t_mini.jsonl # fichier du dossier de données (défaut : pretrain_t2t_mini.jsonl)
  from_subdir: full             # sous-dossier des sorties contenant les poids de départ (--from_weight)
  space:                        # un axe par clé
    muon_lr: [0.005, 0.01]              # liste        → valeurs discrètes
    warmup_ratio: {low: 0, high: 0.1}   # {low, high}  → intervalle (log: true pour une échelle log)
    variante:                           # dict de dicts → variantes nommées (groupes d'arguments)
      dense: {}
      btt4x: {ffn_btt_rank: 6, ffn_btt_mid: "128,64"}

Usage : python study.py etudes/muon_lr.yaml [--set cle=valeur ...]   (--set surcharge `base`)
"""

import argparse
import os
import re
import subprocess
import sys
import time

import optuna
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# Python qui exécute l'entraînement (en local : le venv torch ; dans l'image : le même que l'étude)
TRAIN_PYTHON = os.environ.get("MINIMIND_TRAIN_PYTHON", sys.executable)
OUT_DIR = os.environ.get("MINIMIND_OUT_DIR", os.path.join(HERE, "..", "out"))
VAL_RE = re.compile(r"^step (\d+): val_loss ([\d.]+)")


def suggest(trial, space):
    """Tire les hyperparamètres du trial et retourne les arguments pour train_pretrain.py."""
    args = {}
    for key, spec in space.items():
        if isinstance(spec, list):
            args[key] = trial.suggest_categorical(key, spec)
        elif isinstance(spec, dict) and "low" in spec:
            kind = (
                trial.suggest_int
                if all(isinstance(spec[k], int) for k in ("low", "high"))
                else trial.suggest_float
            )
            args[key] = kind(key, spec["low"], spec["high"], log=spec.get("log", False))
        else:  # variantes nommées : le nom est l'hyperparamètre, la valeur un groupe d'arguments
            args.update(spec[trial.suggest_categorical(key, list(spec))])
    return args


def run_trial(trial, study_cfg, base):
    args = {**base, **suggest(trial, study_cfg["space"])}
    name = f"{study_cfg['name']}-t{trial.number}"
    out = os.path.join(OUT_DIR, f"etude_{study_cfg['name']}")
    os.makedirs(out, exist_ok=True)
    cmd = [
        TRAIN_PYTHON,
        "-u",
        "train_pretrain.py",
        "--save_dir",
        out,
        "--save_weight",
        name,
    ]
    cmd += [
        "--use_mlflow",
        "--mlflow_experiment",
        f"minimind-{study_cfg['name']}",
        "--run_name",
        name,
    ]
    cmd += [x for k, v in args.items() for x in (f"--{k}", str(v))]
    cmd += ["--mlflow_uri", os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh")]
    # chemins des données et des poids de départ : relatifs aux dossiers montés (cluster) ou locaux
    data_dir = os.environ.get("MINIMIND_DATA_DIR", os.path.join(HERE, "..", "dataset"))
    cmd += ["--data_path", os.path.join(data_dir, study_cfg.get("data_file", "pretrain_t2t_mini.jsonl"))]
    if "tokenizer_subdir" in study_cfg:  # tokenizer rangé avec les données (ex. fr_en/tokenizer_fr65_16k)
        cmd += ["--tokenizer_path", os.path.join(data_dir, study_cfg["tokenizer_subdir"])]
    if "from_subdir" in study_cfg:
        cmd += ["--from_dir", os.path.join(OUT_DIR, study_cfg["from_subdir"])]
    trial.set_user_attr("args", args)

    start, value = time.time(), None
    with open(os.path.join(out, f"{name}.log"), "w") as log:
        proc = subprocess.Popen(
            cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            log.write(line)
            log.flush()
            m = VAL_RE.match(line)
            if m:
                value = float(m.group(2))
                trial.report(value, int(m.group(1)))
                if trial.should_prune():
                    proc.terminate()
                    proc.wait()
                    raise optuna.TrialPruned()
        code = proc.wait()
    trial.set_user_attr("minutes", round((time.time() - start) / 60, 1))
    if code != 0 or value is None:
        raise RuntimeError(
            f"trial {name} en échec (code {code}), voir {out}/{name}.log"
        )
    return value


def run_study(study_cfg, overrides=None):
    """Exécute (ou reprend) l'étude. Retourne l'objet optuna.Study."""
    base = {**study_cfg.get("base", {}), **(overrides or {})}
    space = study_cfg["space"]
    sampler_name = study_cfg.get("sampler", "grid")
    if sampler_name == "grid":
        grid = {k: (v if isinstance(v, list) else list(v)) for k, v in space.items()}
        assert all("low" not in v for v in space.values() if isinstance(v, dict)), (
            "grid : pas d'intervalle"
        )
        sampler, n_trials = optuna.samplers.GridSampler(grid), None
    else:
        sampler = {
            "tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
        }[sampler_name](seed=0)
        n_trials = study_cfg["n_trials"]
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=3)
        if study_cfg.get("pruner") == "median"
        else optuna.pruners.NopPruner()
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    study = optuna.create_study(
        study_name=study_cfg["name"],
        storage=f"sqlite:///{os.path.abspath(OUT_DIR)}/optuna.db",
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )
    # un trial resté "en cours" vient d'un processus tué : on le marque en échec pour qu'il soit rejoué
    for t in study.get_trials(states=[optuna.trial.TrialState.RUNNING]):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    study.optimize(
        lambda t: run_trial(t, study_cfg, base),
        n_trials=n_trials,
        catch=(RuntimeError,),
    )
    return study


def summary(study):
    """Tableau markdown des trials, du meilleur au moins bon."""
    done = [t for t in study.trials if t.value is not None]
    lines = [
        "| trial | val_loss | minutes | état | paramètres |",
        "|---|---|---|---|---|",
    ]
    for t in sorted(done, key=lambda t: t.value) + [
        t for t in study.trials if t.value is None
    ]:
        value = f"{t.value:.4f}" if t.value is not None else "—"
        lines.append(
            f"| {t.number} | {value} | {t.user_attrs.get('minutes', '—')} | {t.state.name} | {t.params} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study_file")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="surcharges de `base`, ex. --set max_tokens=1000000",
    )
    cli = parser.parse_args()
    with open(cli.study_file) as f:
        cfg = yaml.safe_load(f)
    study = run_study(cfg, dict(kv.split("=", 1) for kv in cli.set))
    print(summary(study))
