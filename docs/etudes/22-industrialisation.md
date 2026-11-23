# 22 — Industrialisation : traçabilité, tests, registre, évaluation standard

**Question.** Que manquait-il à la chaîne (Dagster, Optuna, MLflow) pour qu'un résultat soit auditable et qu'un
modèle soit livrable ?

## Ce qui a été ajouté

| Brique | Mise en œuvre |
|---|---|
| Traçabilité | chaque run porte le commit du code, l'image, l'empreinte des données et du tokenizer, le GPU (`trainer/provenance.py`) |
| Métriques système | charge et mémoire des GPU, processeur, réseau, enregistrées par MLflow pendant le run |
| Robustesse | une panne du serveur de suivi n'arrête plus l'entraînement : mesures perdues, calcul poursuivi (test automatique) |
| Tests | 10 tests de fumée : un pas, reprise exacte, mode distribué, variantes, serveur de suivi injoignable |
| Intégration continue | GitHub Actions à chaque commit |
| Registre de modèles | `minimind-fr-en-base` et `minimind-fr-en-chat`, alias `reference`, au format Hugging Face, liés à leur run (`scripts/enregistrer_modele.py`) |
| Évaluation standard | lm-evaluation-harness, épreuves françaises et anglaises (`trainer/eval_bench.py`) |
| Calcul externe | Kaggle, Colab, SkyPilot (fiche 20) |

Incidents qui ont motivé ces ajouts : un run de 150M tué par un redémarrage du serveur de suivi (provoqué par
l'envoi d'un fichier de 167 Mo qui dépassait sa mémoire) ; des résultats d'évaluation perdus parce qu'ils
n'étaient écrits qu'en fin de script ; un fichier de configuration Kubernetes contenant un certificat
périmé, toléré par `kubectl` mais refusé par les outils Python.

## Évaluation standard

Précision en %, plus haut = mieux (hasard : 25 %, sauf XNLI 33 % et PIQA 50 %) :

| Épreuve | Ce modèle, 71M | SmolLM2 135M | SmolLM2 360M | CroissantLLM 1,3B |
|---|---|---|---|---|
| Grammaire française | **61,3** | 42,9 | 44,5 | 77,3 |
| Vocabulaire français | **58,0** | 30,3 | 37,0 | 78,2 |
| XNLI (FR) | **39,6** | 35,3 | 37,1 | 48,8 |
| HellaSwag (FR) | 28,1 | 29,1 | 31,8 | 50,3 |
| ARC (FR) | 23,8 | 23,6 | 26,7 | 30,5 |
| Belebele (FR) | 23,3 | 22,3 | 24,6 | 26,8 |
| HellaSwag (EN) | 27,7 | 43,2 | 56,3 | 53,7 |
| SciQ (EN) | 55,4 | 78,6 | 86,5 | 80,0 |

## Conclusion

Le modèle maîtrise la langue française mieux que des modèles publics cinq fois plus gros, mais reste au niveau du
hasard sur le raisonnement et les connaissances, comme eux. La chaîne est désormais auditable de bout en bout :
un chiffre se rattache à un commit, à des données et à un modèle inscrit.

**Reproduire.** `uv run pytest -q tests` ; `trainer/eval_bench.py` ; `scripts/enregistrer_modele.py`.
