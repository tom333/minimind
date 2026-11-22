"""
Script exécuté SUR la machine distante (Kaggle ou Colab). Il n'est pas lancé à la main : scripts/kaggle.sh
ou scripts/colab.sh y injecte la configuration (CONFIG) puis l'envoie.

Déroulé : diagnostic de la machine → clone du dépôt → environnement Python identique au nôtre (uv) →
branchement des données → reprise d'un run interrompu si un point de reprise existe → exécution du mode
demandé. Tout ce qui doit survivre va dans CONFIG["out"] (sortie du notebook Kaggle, dossier Colab ou Drive).
Modes : diagnostic (machine + banc de débit court), bench (banc complet), train (entraînement avec
reprise et arrêt propre avant la limite de durée de la session).
"""

import glob
import os
import shutil
import subprocess
import sys

CONFIG = __CONFIG__

SRC, OUT, INPUT = "/tmp/minimind", CONFIG["out"], CONFIG["input"]  # code et environnement hors de la sortie sauvegardée


def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


def cuda_ok(python):
    return sh(f"{python} -c 'import torch, sys; print(torch.__version__, torch.cuda.is_available()); "
              f"sys.exit(0 if torch.cuda.is_available() and hasattr(torch.optim, \"Muon\") else 1)'", check=False) == 0


# 1. la machine
sh("nvidia-smi", check=False)
sh("nproc; free -g | head -2; df -h /tmp | cat", check=False)

# 2. le code
os.makedirs(OUT, exist_ok=True)
sh(f"rm -rf {SRC}; git clone --depth 1 -b {CONFIG['ref']} {CONFIG['repo']} {SRC}")
os.chdir(SRC)
sh("git log --oneline -1")

# 3. l'environnement : celui du dépôt (uv.lock). Si le pilote de la machine est trop ancien pour les
#    bibliothèques CUDA verrouillées, on réinstalle PyTorch pour une version de CUDA antérieure.
sh("pip install -q uv")
python = f"{SRC}/.venv/bin/python"
sh("uv sync --frozen --no-dev", check=False)
if not cuda_ok(python):
    for index in ("cu128", "cu126"):
        sh(f"uv pip install --python {python} --reinstall torch --index-url https://download.pytorch.org/whl/{index}", check=False)
        if cuda_ok(python):
            break
    else:
        sys.exit("Aucune version de PyTorch avec GPU et optimiseur Muon n'a pu être installée : voir le journal.")

# 4. les données (entraînement seulement) : tokens en cache (.npy), étiquettes de langue et tokenizer
if CONFIG["mode"] == "train":
    caches = glob.glob(f"{INPUT}/**/*.train_rand2000.npy", recursive=True)
    tokenizers = glob.glob(f"{INPUT}/**/tokenizer.json", recursive=True)
    if not caches or not tokenizers:
        sh(f"find {INPUT} -maxdepth 4 | head -50", check=False)
        sys.exit(f"Données introuvables dans {INPUT} : ont-elles été envoyées ?")
    data_path = caches[0][: -len(".train_rand2000.npy")]
    tokenizer_path = os.path.dirname(tokenizers[0])
    print("données :", data_path, "| tokenizer :", tokenizer_path, flush=True)

# 5. reprise : point de sauvegarde d'une session précédente (Kaggle : sortie du notebook attachée en entrée)
os.makedirs(f"{OUT}/resume", exist_ok=True)
for path in glob.glob(f"{INPUT}/**/out/resume/*_resume.pth", recursive=True):
    shutil.copy(path, f"{OUT}/resume/")
    print("point de reprise récupéré :", path, flush=True)

# 6. le travail
gpus = int(CONFIG.get("gpus", 1))
env = ("" if gpus > 1 else "CUDA_VISIBLE_DEVICES=0 ") + "MINIMIND_MLFLOW_URI=" + CONFIG.get("mlflow_uri", "")
# plusieurs GPU sur la machine : entraînement distribué (un lot par carte, gradients moyennés à chaque pas)
launcher = f"{SRC}/.venv/bin/torchrun --standalone --nproc_per_node {gpus}" if gpus > 1 else f"{python} -u"
os.chdir(f"{SRC}/trainer")
if CONFIG["mode"] in ("diagnostic", "bench"):
    only = "--only base,relu2,mod-25,150M" if CONFIG["mode"] == "diagnostic" else ""
    for dtype in CONFIG.get("dtypes", ["bfloat16", "float16"]):
        sh(f"{env} {python} bench_speed.py --dtype {dtype} {only} 2>&1 | grep -v Warning | tee {OUT}/bench_{dtype}.log", check=False)
elif CONFIG["mode"] == "cmd":  # commande libre dans trainer/ avec l'environnement du dépôt (diagnostics)
    sh(f"{env} {python} {' '.join(CONFIG['args'])} 2>&1 | grep -v Warning | tee {OUT}/{CONFIG['name']}.log", check=False)
else:
    args = " ".join(CONFIG["args"])
    try:  # suivi dans MLflow seulement si le serveur est joignable depuis cette machine
        import urllib.request

        urllib.request.urlopen(CONFIG["mlflow_uri"] + "/health", timeout=8)
        args += f" --use_mlflow --mlflow_uri {CONFIG['mlflow_uri']} --mlflow_experiment minimind-kaggle --run_name {CONFIG['name']}"
    except Exception as e:
        print("MLflow injoignable, suivi dans le journal seulement :", type(e).__name__, flush=True)
    code = sh(
        f"{env} {launcher} train_pretrain.py --data_path {data_path} --tokenizer_path {tokenizer_path} "
        f"--save_dir {OUT} --resume_dir {OUT}/resume --save_resume 1 --from_resume 1 "
        f"--max_hours {CONFIG.get('max_hours', 8)} --save_weight {CONFIG['name']} {args} 2>&1 | grep -v Warning | tee -a {OUT}/{CONFIG['name']}.log",
        check=False,
    )
    print("code de sortie de l'entraînement :", code, flush=True)
sh(f"ls -la {OUT} {OUT}/resume; du -sh {OUT}", check=False)
