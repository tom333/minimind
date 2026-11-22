"""
Évaluation sur des bancs standards (lm-evaluation-harness), les mêmes que pour les modèles publics.

Le modèle MiniMind est enveloppé dans l'adaptateur Hugging Face de la librairie (HFLM) : les épreuves à choix
sont notées par vraisemblance (le modèle n'a pas à générer), ce qui convient aux petits modèles.
Les résultats vont dans MLflow (expérience minimind-eval-bench), un run par modèle, avec la traçabilité.

Repère utile : sur une épreuve à N choix, le hasard donne 1/N (25 % pour 4 choix) ; un petit modèle se juge
d'abord par son écart au hasard, ensuite par comparaison aux modèles publics de même taille.

Usage :
  python eval_bench.py moi=../out/etude_fr-en-full/fr-en-full-t0_768.pth smollm2=hf:HuggingFaceTB/SmolLM2-135M
  python eval_bench.py chat=registre:minimind-fr-en-chat@reference --taches francais
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

# Épreuves à choix multiples, notées par vraisemblance. Entre parenthèses : nombre de choix (le hasard = 1/N).
TACHES = {
    "francais": [
        "french_bench_hellaswag",  # suite plausible d'une situation (4)
        "french_bench_arc_challenge",  # questions de sciences niveau scolaire (4)
        "french_bench_grammar",  # grammaire (4)
        "french_bench_vocab",  # vocabulaire (4)
        "belebele_fra_Latn",  # compréhension d'un texte (4)
        "xnli_fr",  # implication entre deux phrases (3)
    ],
    "anglais": ["hellaswag", "arc_easy", "piqa", "sciq", "belebele_eng_Latn"],
}
TACHES["tout"] = TACHES["francais"] + TACHES["anglais"]


def charger(spec, tokenizer_path, device):
    """nom=chemin.pth (poids MiniMind), nom=hf:dépôt (modèle public) ou nom=registre:modèle@alias (registre MLflow)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, path = spec.split("=", 1)
    if path.startswith("hf:"):
        tok = AutoTokenizer.from_pretrained(path[3:])
        model = AutoModelForCausalLM.from_pretrained(path[3:], dtype=torch.bfloat16)
        return name, path, model.to(device).eval(), tok

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    if path.startswith(
        "registre:"
    ):  # modèle inscrit : téléchargé depuis MLflow au format Hugging Face
        import mlflow

        mlflow.set_tracking_uri(
            os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh")
        )
        folder = mlflow.artifacts.download_artifacts(
            f"models:/{path[len('registre:') :]}"
        )
        tok = AutoTokenizer.from_pretrained(folder)
        model = MiniMindForCausalLM.from_pretrained(folder, dtype=torch.float32)
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tok)))
        model.load_state_dict(
            {k: v.float() for k, v in torch.load(path, map_location="cpu").items()}
        )
    return name, path, model.to(device).eval(), tok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modeles",
        nargs="+",
        help="nom=chemin.pth | nom=hf:dépôt | nom=registre:modèle@alias",
    )
    parser.add_argument(
        "--taches",
        default="tout",
        help="francais | anglais | tout | liste séparée par des virgules",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=0,
        help="nombre d'exemples par épreuve (0 = tout ; utile pour un essai)",
    )
    parser.add_argument(
        "--tokenizer_path", default="../dataset/fr_en/tokenizer_fr65_16k"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mlflow_experiment", default="minimind-eval-bench", help="vide = pas d'envoi"
    )
    parser.add_argument("--sortie", default="../out/evals")
    args = parser.parse_args()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tasks = TACHES.get(args.taches) or args.taches.split(",")
    os.makedirs(args.sortie, exist_ok=True)
    table = {}
    for spec in args.modeles:
        name, path, model, tok = charger(spec, args.tokenizer_path, args.device)
        lm = HFLM(
            pretrained=model,
            tokenizer=tok,
            batch_size=args.batch_size,
            max_length=getattr(model.config, "max_position_embeddings", 2048),
        )
        out = simple_evaluate(
            model=lm, tasks=tasks, limit=args.limite or None, log_samples=False
        )
        scores = {}
        for task, res in out["results"].items():
            for key in ("acc_norm,none", "acc,none", "word_perplexity,none"):
                if key in res:
                    scores[task] = (
                        key.split(",")[0],
                        res[key],
                        res.get(key.replace(",", "_stderr,"), None),
                    )
                    break
        table[name] = scores
        with open(os.path.join(args.sortie, f"bench_{name}.json"), "w") as f:
            json.dump(
                {
                    "modele": path,
                    "taches": tasks,
                    "limite": args.limite,
                    "scores": scores,
                },
                f,
                indent=1,
                ensure_ascii=False,
            )
        print(
            f"{name} : " + "  ".join(f"{t} {s[1]:.3f}" for t, s in scores.items()),
            flush=True,
        )
        if args.mlflow_experiment:
            import mlflow

            from trainer.provenance import git_commit

            mlflow.set_tracking_uri(
                os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh")
            )
            mlflow.set_experiment(args.mlflow_experiment)
            with mlflow.start_run(run_name=name):
                mlflow.set_tags(
                    {
                        "git_commit": git_commit(),
                        "modele": path,
                        "limite": str(args.limite),
                    }
                )
                mlflow.log_metrics({f"{t}_{s[0]}": s[1] for t, s in scores.items()})
        del model, lm
        torch.cuda.empty_cache()

    names = list(table)
    print("\n" + f"{'épreuve':30}" + "".join(f"{n:>16}" for n in names))
    for task in tasks:
        cells = [table[n].get(task) for n in names]
        print(
            f"{task:30}"
            + "".join(f"{c[1]:16.3f}" if c else f"{'—':>16}" for c in cells)
        )
