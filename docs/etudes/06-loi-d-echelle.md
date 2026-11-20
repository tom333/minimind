# 06 — Loi d'échelle : que rapporte un token de plus ?

**Question.** Comment la loss évolue-t-elle avec la quantité de données, et peut-on prédire le résultat d'un run long à partir de runs courts ?

**Protocole.** Même modèle, même learning rate, quatre budgets de tokens ; chaque run déroule son propre
calendrier complet. Ajustement de `L(D) = L∞ + B / D^β`, puis vérification sur une epoch entière (332M tokens).
Les quatre runs courts utilisent les réglages d'origine ; le run long utilise la recette finale.

## Résultats

| Tokens | val_loss | Gain depuis le budget précédent |
|---|---|---|
| 12,5M | 5,66 | — |
| 25M | 4,08 | −1,58 |
| 50M | 2,95 | −1,13 |
| 100M | 2,46 | −0,49 |

| Ajustement | Plancher L∞ | Exposant β | Prédiction à 332M tokens |
|---|---|---|---|
| Sur les 4 points | 1,51 | 0,73 | 1,89 |
| Sur les 3 derniers | 2,09 | 1,21 | 2,18 |

Run complet (15 256 pas, recette finale) : **val_loss 2,006**, précision de premier choix 55,2 %.

## Conclusion

Le résultat tombe dans la plage prédite (1,89 à 2,18). Les deux ajustements divergent parce que les points
se situent au tout début de l'apprentissage, là où la loss chute bien plus vite que la loi de puissance
habituelle (β autour de 0,3 dans la littérature). Autre enseignement : la recette améliorée fait gagner
0,60 à 50M tokens, mais ne fait pas sortir le run long de la plage prédite avec les anciens réglages.
Les optimisations accélèrent l'apprentissage ; elles déplacent peu le plancher du modèle.

**Conséquence de méthode.** À 50M tokens, le modèle est limité par les données et non par sa capacité.
Un banc d'essai court convient pour juger une optimisation, pas un ajout de capacité (études 08 et 09).

**Limites.** Quatre points pour trois paramètres ajustés. Runs courts et run long n'utilisent pas la même recette.

**Reproduire.** `trainer/sweep_tokens.sh` ; run complet : étude `memoire_full.yaml` sans mémoire, ou les réglages du fichier `trainer/etudes/fr_en_full.yaml`.
