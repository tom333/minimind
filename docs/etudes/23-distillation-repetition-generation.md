# 23 — Trois leviers hors architecture : distillation, répétition des données, réglages de génération

**Question.** Sans changer la forme du modèle, que rapportent un enseignant, la relecture des mêmes données et
les réglages de génération ?

**Protocole.** Modèle de 71M, recette de référence, 50M tokens par essai. Bruit : 0,012. Plus bas = mieux.
- *Distillation* (Peng et al. 2024 ; Busbridge et al. 2025) : un 71M neuf imite notre 71M complet (1 Md de
  tokens, val_loss 2,967) : 50 meilleures probabilités de l'enseignant, 90 % d'imitation et 10 % de vrais tokens.
- *Répétition* (Muennighoff et al. 2023) : 50M tokens d'entraînement tirés d'un sous-ensemble relu 2 ou 3 fois.
- *Génération* : 60 débuts de textes de test, 96 tokens écrits, jugés par CroissantLLM 1,3B.

## Résultats

| Distillation (corpus web courant) | val_loss | Écart | Durée |
|---|---|---|---|
| Sans enseignant (2 graines) | 3,459 | — | 27 min |
| **Température 1** | **3,364** | **−0,095** | 42 min |
| Température 0,5 | 3,429 | −0,030 | 42 min |

| Répétition (corpus sélectionné) | val_loss | Écart |
|---|---|---|
| 50M tokens tous neufs | 3,383 | — |
| 25M tokens lus 2 fois | 3,410 | +0,027 |
| 16,7M tokens lus 3 fois | 3,476 | +0,093 |

| Génération | 4-grammes répétés | 3-grammes distincts | Perplexité jugée |
|---|---|---|---|
| Texte humain (repère) | 1,6 % | 96,8 % | 4,3 |
| Glouton | 63 % | 35 % | 2,6 |
| min-p 0,05 | 29 % | 68 % | 7,8 |
| top-p 0,9 | 16 % | 80 % | 12,9 |
| **Pénalité de répétition 1,2** | **3,8 %** | **95 %** | **12,1** |
| min-p + DRY | 4,6 % | 91 % | 15,4 |
| Tirage simple, température 0,8 | 7 % | 90 % | 25,4 |
| Aucun 3-gramme répété | 0 % | 100 % | 14,7 |

## Conclusion

- **La distillation rend chaque token plus instructif** : −0,095 à nombre de tokens égal, autant que doubler le
  modèle. Mais l'essai dure 42 min au lieu de 27 (calcul de l'enseignant, lot réduit) : à temps égal, le gain
  disparaît en grande partie. Elle paie quand les données manquent, ou pour réduire un grand modèle.
- **Relire les données coûte plus que ce qu'annonce la littérature** à cette échelle : deux passes sont presque
  gratuites, trois nettement pénalisantes. Le modèle est limité par l'information, pas par le calcul.
- **La pénalité de répétition 1,2 est adoptée par défaut** : elle supprime les boucles en gardant le texte le
  plus plausible parmi les méthodes qui les suppriment.

**Limites.** Une graine par essai. Enseignant de même taille que l'élève. La perplexité jugée dépend du juge.

**Reproduire.** Études `distillation.yaml`, `repetition.yaml` ; `trainer/eval_generation.py`.
