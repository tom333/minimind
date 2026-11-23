# 17 — Accélérer l'entraînement : réglages système, FFN, profondeur, à temps égal

**Question.** Peut-on diviser par deux le temps d'entraînement, pour faire tenir un modèle de 150M en 8 heures
sur la RTX 3060 ? Un gain de vitesse ne compte que s'il ne coûte pas plus en loss qu'il ne rapporte en tokens.

**Protocole.** Trois niveaux de mesure. (1) Banc de débit (`trainer/bench_speed.py`) : temps d'un pas isolé.
(2) Essais de 50M tokens : loss à budget de tokens égal. (3) **Essais à temps égal** : chaque variante reçoit le
nombre de tokens qu'elle traite dans le temps où la référence en traite 200M. Bruit : 0,012.

## Résultats

Réglages système (banc ; référence 599 ms par pas, 36 300 tokens/s) : calculs fp32 accélérés +0,3 %, AdamW
fusionné +0,4 %, compilation « max-autotune » −0,3 %, sans plafonnement du gradient +0,6 %, Muon à 3 itérations
+5,2 %. **Aucun gain exploitable.**

Variantes du modèle à 50M tokens (référence 3,462) :

| Variante | Paramètres | val_loss | Débit |
|---|---|---|---|
| SwiGLU, 8 blocs (référence) | 71,3M | 3,462 | — |
| ReLU² à largeur égale | 56,3M | 3,495 | +24 % |
| ReLU² à paramètres égaux | 71,3M | 3,472 | +2 % |
| 10 blocs, FFN plus étroit | 71,3M | 3,447 (3,448 sur 2 graines) | 0 % en conditions réelles |
| 12 blocs / 16 blocs | 70,7M / 71,3M | 3,445 / 3,443 | plus lent |
| FFN unique partagé par les 8 blocs | 32,1M | 3,597 | 0 % |

**À temps de calcul égal** (200M tokens pour la référence) :

| Variante | Tokens lus | val_loss |
|---|---|---|
| Référence | 200M | 3,120 |
| ReLU² à largeur égale | 248M | 3,121 |
| 10 blocs | 211M | 3,100 (6 % de temps en plus) |
| Mixture-of-Depths 25 % | 281M | 3,151 |

## Conclusion

Sur cette carte, il n'y a pas d'accélération gratuite : le pas est déjà optimisé, et à temps égal aucune
variante ne bat le modèle d'origine. ReLU² donne la même qualité dans un modèle 20 % plus léger, utile quand la
mémoire compte. La profondeur aide un peu à paramètres égaux, mais coûte du temps. Le banc de débit surestime
certains gains (« 10 blocs » : +5,6 % au banc, nul en réel) : seuls les débits mesurés en entraînement font foi.
L'objectif « 150M en 8 h » ne s'atteint pas par l'architecture seule : il passe par le matériel (fiche 20).

**Limites.** Une graine pour la plupart des variantes ; l'étude à temps égal porte sur un seul budget.

**Reproduire.** `trainer/bench_speed.py` ; études `ffn.yaml`, `profond.yaml`, `ffn_partage.yaml`, `graines.yaml`, `regime.yaml`.
