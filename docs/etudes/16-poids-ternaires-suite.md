# 16 — Poids ternaires, suite : recette, techniques publiées et agrandissement

**Question.** L'écart de la fiche 11 (+8 % de loss à taille égale) se réduit-il avec une recette adaptée,
avec les techniques publiées (BitNet b1.58 2B4T, Ternary Bonsai 2), ou en agrandissant le modèle ?

**Protocole.** Corpus bilingue, 50M tokens, poids ternaires par ligne, même graine. Référence 16 bits au même
budget : 3,462. Bruit : 0,012. Plus bas = mieux. La part de poids à zéro est mesurée à chaque run.

## Résultats

| Variante | val_loss | Écart avec la référence ternaire | Source de l'idée |
|---|---|---|---|
| Recette du 16 bits (référence ternaire) | 3,593 | — | fiche 11 |
| Échelle par groupe de 128 poids | 3,595 | +0,002 | Ternary Bonsai 2 |
| Groupes de 128 + rotation de Hadamard de l'entrée | 3,595 | +0,002 | Ternary Bonsai 2 |
| SubLN + FFN ReLU² | 3,605 | +0,011 | BitNet b1.58 2B4T |
| 16 bits pendant 90 %, puis bascule progressive | 3,637 | +0,043 | ParetoQ, Nielsen et al. 2025 |
| **Taux d'apprentissage ×2** | **3,562** | **−0,032** | BitNet |
| Taux ×3 | 3,588 | −0,006 | — |
| Taux ×4 | 3,606 | +0,013 | — |

Un tiers des poids vaut zéro dans tous les essais (31,6 à 33,5 %).

Agrandissement à mémoire inférieure (taux ×2) :

| Modèle | Taille en mémoire | val_loss |
|---|---|---|
| 16 bits, 71M | 143 Mo | 3,462 |
| Ternaire, 71M | ~40 Mo | 3,562 |
| **Ternaire, 104M** (largeur 896, 9 blocs) | **~50 Mo** | **3,512** |

## Conclusion

Le seul réglage qui compte est le taux d'apprentissage, doublé : l'écart avec le 16 bits passe de +0,131 à
+0,100. C'est le conseil de BitNet, et il tient avec l'optimiseur Muon, ce qu'aucune publication trouvée n'avait
mesuré. Les techniques de Bonsai (groupes, rotation) servent à *compresser* un modèle déjà entraîné, dont les
poids ont des valeurs extrêmes ; appris directement en ternaire, les poids n'en ont pas, et elles n'apportent
rien. SubLN et ReLU² sont neutres à cette taille. La bascule tardive est pénalisée par le budget court (5M
tokens pour s'adapter) et reste à juger sur un run long.

Agrandir de 46 % comble la moitié de l'écart, pour un modèle presque trois fois plus léger que le 16 bits. La
littérature (Spectra, 2024) indique que l'écart se réduit lentement avec la taille : l'idée « plus gros en
ternaire, à mémoire égale » se vérifie dans le sens attendu, et demande l'échelle suivante pour aboutir.

**Limites.** Une graine par variante ; budget court, défavorable au modèle agrandi. L'embedding reste en 16 bits
et pèse plus que tous les blocs ternaires (vocabulaire de 16 000) : le passer en 4 ou 8 bits est la prochaine
étape pour un comptage honnête de la mémoire.

**Reproduire.** `--ternary row|group --ternary_hadamard 1 --ternary_start 0.9 --subln 1` ; études
`ternaire_v2.yaml`, `ternaire_lr.yaml`, `ternaire_grand.yaml`.
