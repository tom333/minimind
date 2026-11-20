# 08 — Matrices structurées dans le FFN

**Question.** Remplacer les grandes matrices pleines du FFN par des matrices structurées permet-il, à paramètres et calcul théorique égaux, un FFN plus large et un meilleur modèle ? (Qiu et al., 2024, « Compute Better Spent »)

**Principe.** Une matrice Block Tensor-Train à deux noyaux remplace `entrée → sortie` par deux étages de
petites matrices. À coût égal en paramètres et en opérations, la largeur interne du FFN passe de 2 432 à
4 864. Muon est appliqué bloc par bloc (`trainer/muon.py`), combinaison non documentée à ma connaissance.

**Protocole.** 50M tokens, même batch effectif, recette finale. Référence à FFN plein : 2,338. Bruit : 0,012.

## Résultats

| Mesure | FFN plein | FFN structuré, largeur ×2 | Écart | Verdict |
|---|---|---|---|---|
| Paramètres | 63,9M | 63,3M | égal | — |
| val_loss | 2,338 | 2,396 | +0,058 | moins bon : 5 fois le bruit |
| Loss sur les tokens rares | 3,343 | 3,437 | +0,094 | moins bon, c'est là que l'écart est le plus fort |
| Mémoire GPU (batch de 16) | 3,8 Go | 9,4 Go | ×2,5 | mauvais : le batch de 32 ne tient plus dans 12 Go |
| Durée d'une epoch (batch de 16) | 193 min | 472 min | ×2,4 | mauvais |

La variante de largeur ×3,4 ne tient pas en mémoire.

## Conclusion

Résultat négatif. L'égalité annoncée porte sur le nombre d'opérations, mais la couche structurée passe par
un état intermédiaire d'environ 12 000 valeurs par token, contre 2 400 pour le FFN plein, qu'il faut garder
en mémoire pour le gradient. À paramètres et calcul théorique égaux, elle coûte 2,5 fois plus de mémoire et
2,4 fois plus de temps. Sur cette carte, la piste est écartée avant même de considérer la qualité.

**Limites.** Un seul run. L'initialisation et le learning rate n'ont pas été réglés pour cette structure,
alors que les auteurs insistent sur ce point. Budget court, défavorable à un ajout de largeur.

**Reproduire.** `--ffn_btt_rank 9 --ffn_btt_mid 76,64` ; étude `trainer/etudes/btt.yaml` ; classe `BTTLinear` dans `model/feedforward.py`.
