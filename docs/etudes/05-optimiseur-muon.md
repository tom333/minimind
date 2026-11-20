# 05 — Optimiseur : Muon contre AdamW

**Question.** L'optimiseur Muon, présenté comme le gain le plus robuste du « speedrun » nanoGPT, tient-il ses promesses sur un modèle de 64M de paramètres ?

**Principe.** Pour chaque matrice de poids, Muon remplace la mise à jour par sa version orthogonalisée :
toutes les directions de la matrice avancent avec la même amplitude, au lieu de laisser quelques directions
dominantes tout absorber. L'orthogonalisation est approchée par cinq itérations de Newton-Schulz.
Ici, Muon traite les 56 matrices des blocs (59M de paramètres) ; AdamW garde l'embedding et les normes.

**Protocole.** 50M tokens, batch de 64 séquences, warmup 2 %, weight decay 0,1. Référence AdamW : 2,486
(moyenne de deux graines). Bruit : 0,012. Plus bas = mieux. Une implémentation écrite à la main
(`trainer/muon.py`) a été comparée à `torch.optim.Muon`.

## Résultats

| Réglage | val_loss | Écart avec AdamW | Verdict |
|---|---|---|---|
| AdamW (référence) | 2,486 | — | — |
| Muon, LR 0,005 | 2,382 | −0,104 | bon |
| **Muon, LR 0,01** | **2,347 / 2,340** | **−0,143** | le meilleur, sur deux graines |
| Muon, LR 0,02 | 2,355 / 2,351 | −0,133 | bon |
| Muon, LR 0,04 | 2,410 | −0,076 | LR trop haut |

Coût : l'orthogonalisation occupe 15 % du temps d'un pas (mesure de l'étude 01).

Vérification de l'implémentation maison : AdamW identique à PyTorch (écart 6e-8) ; Muon de même qualité
d'orthogonalisation (similarité avec le résultat exact : 0,9263 contre 0,9265).

## Conclusion

Muon fait gagner 0,14 de val_loss, soit plus de dix fois le bruit, pour 15 % de temps en plus. À temps
égal, AdamW aurait besoin d'environ deux fois plus de tokens pour le rattraper. C'est la seule technique
issue de la littérature récente qui ait tenu dans cette configuration. Il gagne aussi en ajustement fin
(étude 12), à condition de baisser son learning rate.

**Limites.** Deux graines pour le meilleur réglage, une seule pour les autres. Les autres optimiseurs
récents (SOAP, Schedule-Free) n'ont pas encore été comparés.

**Reproduire.** `--optimizer muon --muon_lr 0.01` ; `trainer/sweeps/muon.txt`, `muon2.txt` ; commit `b398a00`.
