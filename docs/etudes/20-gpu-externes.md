# 20 — GPU externes : Kaggle, Colab, entraînement distribué et SkyPilot

**Question.** Les GPU gratuits (Kaggle, Colab) et loués peuvent-ils prolonger la RTX 3060 sans perte de qualité
ni de reproductibilité ?

**Mise en place.** Un même script (`cloud/run.py`) s'exécute sur la machine distante : il clone le dépôt, recrée
l'environnement exact (`uv.lock`), branche les données, reprend au dernier point de sauvegarde et s'arrête
proprement avant la fin d'une session (`--max_hours`). Lanceurs : `scripts/kaggle.sh`, `scripts/colab.sh` ; pour
les GPU loués, une tâche SkyPilot (`sky/entrainement.yaml`), validée sur notre propre cluster Kubernetes.

## Résultats

| Mesure | Résultat | Repère |
|---|---|---|
| Débit d'une T4, fp16 (71M) | 21 000 à 21 500 tokens/s | RTX 3060 en bf16 : 36 300 (la T4 fait 58 %) |
| bf16 sur T4 | mémoire insuffisante, même à 71M | — |
| **Loss en fp16 sur T4** (50M tokens) | **3,4607** | bf16 sur RTX 3060 : 3,4620 (écart 0,001, dix fois sous le bruit) |
| Reprise après coupure | identique pas à pas au run ininterrompu, y compris en distribué | — |
| Deux T4 de Kaggle en distribué | 27 000 tokens/s (×1,28) | ×1,75 espéré |

L'entraînement distribué a révélé deux défauts du code d'origine qui auraient gâché un run long sans le faire
planter : le nombre de pas était calculé pour une seule carte (calendrier du taux d'apprentissage arrêté à
mi-course), et l'arrêt programmé était décidé carte par carte (blocage à la limite de durée).

Obstacles rencontrés : sur Kaggle, le GPU et Internet ne sont accordés qu'après vérification du téléphone, et
l'accès Internet doit être activé une première fois dans l'interface ; sur Colab gratuit, l'envoi de fichiers
par la ligne de commande est lent (2 Go impossibles en 90 min) et l'interface web ne se rattache pas à une
session créée en ligne de commande : les données passent par un lien de partage Google Drive.

## Conclusion

Les GPU gratuits sont utilisables en fp16 sans perte de qualité. Une T4 vaut un peu plus de la moitié de la
RTX 3060 ; les deux T4 de Kaggle s'en approchent. C'est une seconde voie pour les runs longs (30 h de GPU par
semaine), pas une accélération. Le gain d'échelle viendra d'un GPU loué, que la tâche SkyPilot permet de
lancer sans rien réécrire.

**Limites.** Gain du distribué limité, probablement par le lot de 32 par carte et l'absence de liaison rapide
entre les cartes ; non étudié plus avant.

**Reproduire.** `scripts/kaggle.sh`, `scripts/colab.sh`, `cloud/run.py`, `sky/entrainement.yaml` ; test du mode
distribué dans `tests/test_entrainement.py`.
