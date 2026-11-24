# 24 — Tables de n-grammes à l'entrée

**Question.** Peut-on ajouter des paramètres qui sont LUS au lieu d'être multipliés, donc presque gratuits en
calcul ? (embeddings hachés ; N-Grammer, Roy et al. 2022)

**Principe.** En plus de son embedding, chaque position reçoit un vecteur lu dans une table, à la ligne donnée
par le hachage de ses 2, 3 ou 4 derniers tokens (262 144 lignes de 64 nombres par ordre). La position ne lit
jamais le futur (test dédié).

**Protocole.** 71M + tables, 50M tokens, même recette et même graine que la référence. Bruit : 0,012.

## Résultats

| Essai | Paramètres ajoutés | val_loss | Écart | Durée |
|---|---|---|---|---|
| Sans tables (2 graines) | — | 3,459 | — | 27 min |
| Paires seules | +17M | 3,441 | −0,018 | 27,9 min |
| Paires, triplets, quadruplets | +50M | 3,441 | −0,018 | 27,7 min |
| Idem, tables apprises 10 fois plus vite | +50M | 3,454 | −0,005 | 28,2 min |
| 150M dense (repère de taille) | +79M | 3,375 | −0,084 | pas deux fois plus long |

## Conclusion

Le coût est bien nul (−1 % de débit), mais le gain est petit : 1,5 fois le bruit. Les triplets et quadruplets
n'ajoutent rien, et l'avance fond pendant l'entraînement (−0,032 au pas 250, −0,021 à la fin) : c'est un départ
plus rapide, pas un meilleur modèle. Deuxième résultat dans ce sens après la couche mémoire (étude 09) : à cette
échelle, des paramètres ajoutés sans calcul ne rapportent presque rien.

**Limites.** Une graine. Tables petites : le record actuel du concours nanoGPT en utilise une de 84,6M lignes,
300 fois plus grande. Les tables sont apprises par gradient ; l'étude 26 mesure ce que donne le comptage direct.
La génération avec cache n'est pas gérée.

**Reproduire.** `--ngram_orders 2,3,4` ; étude `ngrammes.yaml` ; `model/ngram.py`.
