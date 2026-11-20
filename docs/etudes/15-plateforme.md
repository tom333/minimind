# 15 — La plateforme d'expérimentation

**Question.** Comment enchaîner des dizaines d'entraînements sur un seul GPU partagé, sans perdre de résultats ni de traçabilité ?

## Ce qui a été construit

| Besoin | Réponse | Fichiers |
|---|---|---|
| Décrire une campagne d'essais | fichier YAML : espace de recherche, arguments communs, données, poids de départ | `trainer/etudes/*.yaml` |
| Exécuter les essais | Optuna (grille, aléatoire ou recherche guidée), un sous-processus par essai, arrêt anticipé possible | `trainer/study.py` |
| Reprendre après une interruption | base SQLite des études : relancer une étude la reprend là où elle s'est arrêtée | `out/optuna.db` |
| Suivre les mesures | MLflow : paramètres, courbes, évaluation détaillée de fin de run | serveur du cluster |
| Orchestrer | Dagster : un job par étude, des assets pour les données | `orchestration/definitions.py` |
| Partager le GPU | classe de priorité Kubernetes : le run évince le service d'inférence, qui revient seul à la fin | dépôt d'infrastructure |
| Reproduire | image Docker unique pour l'entraînement et l'orchestration | `Dockerfile` |

## Faits marquants

| Constat | Détail |
|---|---|
| Arbitrage du GPU sans intervention | le service d'inférence est replanifié en moins de 15 secondes après la fin d'un run |
| Échecs isolés | une étude dont les essais débordent la mémoire se termine proprement, avec la cause lisible |
| Écart entre poste local et cluster | six allers-retours pour le premier run : chemins d'import sous Dagster, version de pyarrow, cache sur un disque plein, adresse du serveur de suivi |
| Run de 2 heures perdu | processus tué par le redémarrage du service hôte, sans point de reprise : motivation de la reprise par Optuna et du passage aux pods |

## Conclusion

Le passage de scripts shell à cette chaîne a été déclenché par deux incidents concrets : des conflits de
mémoire GPU avec un autre service, et un run long perdu. La mise au point a montré qu'un environnement
identique entre le poste local et le cluster est la condition du reste, d'où la migration prévue vers un
fichier de verrouillage unique.

**Limites.** Un seul nœud GPU. Pas encore de test automatique de la chaîne. Les plages horaires du service
d'inférence sont évitées à la main.

**Reproduire.** `scripts/publier-code-dagster.sh`, `scripts/lancer-etude.sh <fichier>`, `scripts/lancer-job.sh <job>`.
