# 03 — Learning rate, warmup, weight decay et retouches de la littérature

**Question.** Les réglages d'optimisation d'origine sont-ils bons, et les retouches réputées utiles ailleurs le sont-elles ici ?

**Protocole.** 50M tokens par run, tout le reste fixé. Repère de bruit : 0,055 avec la recette d'origine,
0,012 avec la recette finale (étude 02). Plus bas = mieux.

## Résultats

Learning rate, optimiseur AdamW (premier jeu de validation, valeurs comparables entre elles seulement) :

| LR | 1e-4 | 3e-4 | 5e-4 (origine) | 1e-3 | 2e-3 |
|---|---|---|---|---|---|
| val_loss | 5,21 | 3,83 | 3,38 | 3,37 | 4,32 |

Warmup et weight decay (référence 2,950, bruit 0,055) :

| Variante | val_loss | Écart | Verdict |
|---|---|---|---|
| Warmup sur 2 % des pas | 2,915 | −0,035 | dans le bruit, courbe constamment en dessous : retenu |
| Weight decay 0,1 (au lieu de 0,01) | 2,940 | −0,010 | dans le bruit ; retenu par convention |
| Warmup + LR 1,5e-3 | 2,939 | −0,011 | dans le bruit |
| Warmup + LR 2e-3 | 3,206 | +0,256 | nettement moins bon |

Retouches issues du « speedrun » nanoGPT (référence 2,345 avec Muon, bruit 0,012) :

| Variante | val_loss | Écart | Verdict |
|---|---|---|---|
| Calendrier WSD + plafonnement des logits + initialisation à zéro | 2,419 / 2,414 | +0,073 | moins bon, sur deux graines |
| Plafonnement des logits + initialisation à zéro, calendrier cosinus | 2,367 / 2,363 | +0,020 | légèrement moins bon |

## Conclusion

Le learning rate d'origine était déjà sur un optimum plat (5e-4 à 1e-3). Aucune des trois retouches
publiées n'améliore le modèle ; le calendrier WSD est le plus pénalisant à budget fixe. L'architecture
contient déjà des stabilisateurs équivalents (normalisation des requêtes et des clés, embeddings partagés).

**Limites.** Le premier balayage a été jugé sur l'ancien jeu de validation. Les retouches n'ont pas toutes
été testées isolément : le plafonnement et l'initialisation à zéro ne le sont qu'ensemble.

**Reproduire.** `trainer/sweep_lr.sh`, `trainer/sweep_optim.sh`, `trainer/sweeps/retouches*.txt` ; options
`--warmup_ratio`, `--weight_decay`, `--lr_schedule`, `--logit_softcap`, `--zero_init` ; commit `203aff4`.
