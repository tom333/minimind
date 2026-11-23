# 19 — Vers 150M paramètres : premiers essais et coût

**Question.** La marche suivante (150M) vaut-elle son coût, et combien de temps prendrait-elle sur la RTX 3060 ?

**Protocole.** 148M paramètres (largeur 1 024, 10 blocs), même recette que le 71M (taux d'apprentissage non
réajusté), 50M tokens. La variante dense ne tient en mémoire qu'avec un lot de 32 cumulé deux fois (même lot
global). Bruit : 0,012.

## Résultats

| Modèle | val_loss | Français / anglais | Durée (50M tokens) |
|---|---|---|---|
| 71M (référence) | 3,462 | 3,297 / 3,763 | 27 min |
| **150M dense** | **3,375** | 3,213 / 3,670 | 60 min |
| 150M Mixture-of-Depths 25 % | 3,436 | 3,272 / 3,735 | 42 min |

Projection pour 1 milliard de tokens sur la RTX 3060, d'après ces durées réelles : 150M dense ~20 h, 150M avec
routage ~14 h. Sur une T4 (Kaggle, Colab), le banc donne 10 700 tokens/s pour le 150M avec routage.

## Conclusion

Le 150M bat déjà le 71M de 0,087 (sept fois le bruit) avec seulement 0,3 token par paramètre et une recette
non réglée pour lui : la marche suivante paiera. À temps égal sur ce budget court, le petit modèle reste devant,
ce qui est attendu : un gros modèle n'est rentable qu'au-delà d'un certain volume de données. L'objectif de 8 h
par milliard de tokens n'est pas atteignable sur cette carte seule.

**Limites.** Une graine ; budget court. Un essai à 200M tokens est en cours.

**Reproduire.** Études `grand_150m.yaml`, `grand_150m_200.yaml`.
