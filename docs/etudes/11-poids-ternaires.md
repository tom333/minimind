# 11 — Poids à trois valeurs (−1, 0, +1)

**Question.** Que perd-on à n'autoriser que trois valeurs par poids, et vaut-il mieux convertir un modèle entraîné ou l'entraîner ternaire dès le départ ? (Ma et al., 2024, BitNet b1.58)

**Principe.** Chaque matrice des blocs garde des poids précis pour l'apprentissage, mais le modèle calcule
toujours avec leur version arrondie à −1, 0 ou +1, multipliée par une échelle (par matrice ou par ligne).
Le gradient traverse l'arrondi comme s'il était l'identité. L'embedding reste en 16 bits.

**Protocole.** Trois voies comparées sur le même jeu de validation, même recette, même graine :
arrondi brut d'un modèle entraîné ; réentraînement de ce modèle avec l'arrondi (50M puis 150M tokens) ;
entraînement ternaire depuis zéro sur une epoch (332M tokens). Bruit : 0,012. Plus bas = mieux.

## Résultats

| Modèle | val_loss | Écart avec le modèle 16 bits | Précision | Loss sur les tokens rares |
|---|---|---|---|---|
| 16 bits (référence, 332M tokens) | 2,006 | — | 55,2 % | 2,80 |
| Arrondi brut, échelle par matrice | 5,654 | +3,65 | — | — |
| Arrondi brut, échelle par ligne | 5,544 | +3,54 | — | — |
| Réentraîné 50M tokens, échelle par matrice | 2,237 | +0,231 | 51,9 % | 3,14 |
| Réentraîné 50M tokens, échelle par ligne | 2,231 | +0,225 | 52,0 % | 3,13 |
| Réentraîné 150M tokens, échelle par ligne | 2,163 | +0,157 | 52,9 % | 3,02 |
| **Ternaire depuis zéro, 332M tokens** | **2,165** | **+0,159** | 52,9 % | 3,03 |

Pendant l'entraînement depuis zéro, l'écart avec le modèle 16 bits reste stable : +0,129 au pas 4 000,
+0,144 au pas 12 000, +0,159 à la fin. Aucun surcoût de temps (121 minutes par epoch contre 117 à 127).

| Version | Taille | Rapport |
|---|---|---|
| Tout en 16 bits | 128 Mo | 1 |
| Blocs ternaires, embedding en 16 bits | ~21,5 Mo | ÷6 |

## Conclusion

Un modèle six fois plus petit pour 8 % de loss en plus (2,165 contre 2,006). L'arrondi brut détruit le
modèle, mais l'apprentissage avec arrondi récupère 96 % de la perte. Les deux voies convergent vers la
même valeur, 2,163 et 2,165 : la différence est dans le bruit. Cette coïncidence suggère que ~0,16 est le
prix de la capacité perdue en passant à trois valeurs, et non un défaut de méthode. La voie depuis zéro
est la moins coûteuse (332M tokens contre 482M au total). Les tokens rares paient le plus : +0,23 contre
+0,10 pour les tokens fréquents.

**Limites.** Un seul run par variante : le chiffre de 8 % est à confirmer sur plusieurs graines.
Seuls les poids sont arrondis, pas les activations, contrairement à BitNet. Le gain de taille n'est pas
un gain de vitesse sur GPU ; il le serait avec des noyaux adaptés, sur processeur.

**Reproduire.** `--ternary row` ; études `ternaire_qat.yaml`, `ternaire_qat_long.yaml`, `ternaire_scratch.yaml` ; `model/ternary.py` ; commit `53b9b0a`.
