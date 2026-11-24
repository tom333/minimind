"""
À partir de quel moment le résultat d'un essai est-il déjà joué ?

Analyse rétrospective de tous les essais enregistrés dans MLflow. Pour chaque paire d'essais comparables (même
étude, ou même corpus et même budget pour le corpus bilingue), on compare l'écart de val_loss observé à une
fraction f de l'entraînement à l'écart final :
  - accord de signe : l'essai en tête à la fraction f est-il encore en tête à la fin ? (paires dont l'écart final
    dépasse deux fois le bruit, soit 0,024) ;
  - prévision, sur les paires serrées : écart final ≈ β·écart à f ; β < 1 = les avances précoces fondent ;
    l'erreur résiduelle dit avec quelle précision on connaît l'écart final sans finir l'essai.
Les courbes sont lues à fraction d'entraînement égale (interpolation entre deux évaluations), ce qui compare
des essais au même nombre de tokens même quand leur taille de lot diffère.

Usage : python diag_prediction.py [--rafraichir]
"""

import argparse
import itertools
import json
import os

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "..", "out", "evals", "historique_courbes.json")
NOISE = 0.012
AUTRE_MESURE = {"diffusion", "hybride"}  # leur val_loss ne mesure pas la même chose d'un essai à l'autre


def download():
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MINIMIND_MLFLOW_URI", "https://mlflow.tgu.ovh"))
    client, rows = mlflow.MlflowClient(), []
    for exp in client.search_experiments():
        if not exp.name.startswith("minimind-"):
            continue
        for run in client.search_runs([exp.experiment_id], max_results=1000):
            if run.info.status != "FINISHED" or "val_loss" not in run.data.metrics:
                continue
            curve = sorted({m.step: m.value for m in client.get_metric_history(run.info.run_id, "val_loss")}.items())
            if len(curve) >= 4:
                rows.append(dict(etude=exp.name[len("minimind-") :], essai=run.info.run_name, params=dict(run.data.params), courbe=curve))
    json.dump(rows, open(CACHE, "w"))
    return rows


def at(curve, f):
    """val_loss à la fraction f de l'essai (None avant la première évaluation)."""
    steps, values = np.array(curve).T
    x = f * steps[-1]
    return None if x < steps[0] else float(np.interp(x, steps, values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rafraichir", action="store_true", help="relit MLflow au lieu du fichier local")
    args = parser.parse_args()
    rows = download() if args.rafraichir or not os.path.exists(CACHE) else json.load(open(CACHE))
    rows = [r for r in rows if r["etude"] not in AUTRE_MESURE]
    budget = lambda r: (os.path.basename(r["params"].get("data_path", "")), r["params"].get("max_tokens"), r["params"].get("max_steps"), r["params"].get("task"))
    pairs = set()
    for a, b in itertools.combinations(range(len(rows)), 2):
        same_study = rows[a]["etude"] == rows[b]["etude"]
        bilingual = budget(rows[a])[0] == "pretrain_fr65.jsonl" and budget(rows[a])[1] not in (None, "0")
        if budget(rows[a]) == budget(rows[b]) and (same_study or bilingual):
            pairs.add((a, b))
    print(f"{len(rows)} essais, {len(pairs)} paires comparables ; bruit entre deux graines : {NOISE}")
    print(f"\n{'fraction':>9}{'paires':>8}{'accord de signe':>17}{'serrées':>10}{'nettes':>9}{'si avance > 0,024':>19}{'β':>7}{'erreur (serrées)':>18}")
    for f in (0.1, 0.2, 0.3, 0.5, 0.75):
        early, final = [], []
        for a, b in pairs:
            ea, eb = at(rows[a]["courbe"], f), at(rows[b]["courbe"], f)
            if ea is not None and eb is not None:
                early.append(ea - eb)
                final.append(rows[a]["courbe"][-1][1] - rows[b]["courbe"][-1][1])
        early, final = np.array(early), np.array(final)
        decisive = np.abs(final) >= 2 * NOISE
        agree = np.sign(early) == np.sign(final)
        close, clear = decisive & (np.abs(final) < 0.1), np.abs(final) >= 0.1
        trusted = decisive & (np.abs(early) >= 2 * NOISE)  # la lecture précoce est elle-même au-dessus du bruit
        beta = (early[close] @ final[close]) / (early[close] @ early[close])
        resid = np.sqrt(np.mean((final[close] - beta * early[close]) ** 2))
        print(f"{f:9.0%}{decisive.sum():8d}{agree[decisive].mean():17.1%}{agree[close].mean():10.1%}{agree[clear].mean():9.1%}{agree[trusted].mean():19.1%}{beta:7.2f}{resid:18.3f}")
    print("\nserrées : écart final entre 0,024 et 0,10 ; nettes : au-delà de 0,10. Hasard : 50 %.")
