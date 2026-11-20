# 04 — Taille de batch : combien de mises à jour pour un budget donné ?

**Question.** À nombre de tokens fixe, vaut-il mieux de gros batches peu nombreux ou de petits batches nombreux ?

**Protocole.** 50M tokens, AdamW, warmup 2 %, weight decay 0,1. Séquences de 340 tokens, batch de 32
multiplié par un nombre de pas d'accumulation. Deux graines par réglage. Plus bas = mieux.

## Résultats

| Batch effectif | Mises à jour sur 50M tokens | val_loss (graines 42 / 2) | Écart avec le réglage d'origine |
|---|---|---|---|
| 256 séquences (origine) | ~574 | 2,919 / 2,895 | — |
| 128 séquences | ~1 150 | 2,599 / 2,599 | −0,31 |
| 64 séquences | ~2 300 | 2,488 / 2,484 | −0,42 |
| 32 séquences (une seule graine) | ~4 600 | 2,446 | −0,46 |

Le bruit entre graines est de 0,024 au plus : tous les écarts sont très au-delà.

## Conclusion

C'est le gain le plus important de toute la série : −0,42 de val_loss en divisant le batch par quatre,
sans coût supplémentaire. Avec 574 mises à jour, le modèle voyait beaucoup de données mais apprenait trop
rarement. Le rendement décroît ensuite : passer de 64 à 32 séquences ne rapporte plus que 0,04. Le batch de
64 est retenu, car il reste efficace sur le GPU.

**Limites.** Résultat établi à 50M tokens ; sur un run long, le batch optimal est probablement plus grand.
Le learning rate n'a pas été réajusté pour chaque taille de batch.

**Reproduire.** `trainer/sweeps/batch.txt` (`bash sweep.sh batch`) ; option `--accumulation_steps`.
