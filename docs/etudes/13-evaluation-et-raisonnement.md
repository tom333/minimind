# 13 — Mesurer autre chose que la moyenne : évaluation détaillée et raisonnement

**Question.** Que cache la loss moyenne, et le modèle sait-il enchaîner deux étapes de raisonnement ?

**Protocole.** Évaluation détaillée (`trainer/eval_detail.py`) : loss par source du corpus, par classe de
fréquence du token à prédire, précision de premier choix, divergence avec un modèle de référence.
Banc de raisonnement (`trainer/eval_reasoning.py`) : problèmes d'arithmétique générés par programme,
classés par nombre d'opérations enchaînées, 30 problèmes par case, décodage déterministe.

## Résultats

Loss selon la fréquence du token à prédire (modèle du run complet, plus bas = mieux) :

| Classe de tokens | Part des occurrences | Loss |
|---|---|---|
| Fréquents | 50 % | 1,40 |
| Moyens | 40 % | 2,42 |
| Rares | 10 % | 2,80 |
| Moyenne affichée | — | 2,01 |

Deux modèles de loss voisine diffèrent sur 33 à 35 % de leurs prédictions. Ce taux n'est pas encore
étalonné : il manque la mesure entre deux graines de la même recette.

Banc de raisonnement, modèle après ajustement (plus haut = mieux, 100 % attendu) :

| Opérations enchaînées | Chinois | Anglais |
|---|---|---|
| 1 | 17 % | 3 % |
| 2 | 10 % | 0 % |
| 3 | 7 % | 3 % |

Effet du mode « réflexion » (le modèle écrit un raisonnement avant de répondre), 90 tirages :

| Mode | Bonnes réponses |
|---|---|
| Sans réflexion | 10 % |
| Avec réflexion | 3 % |

Question piège (« je dois laver ma voiture à la station à 50 mètres, j'y vais à pied ou en voiture ? ») :
ni notre modèle ni le modèle officiel ne relève que la voiture doit être sur place.

## Conclusion

La moyenne masque un écart du simple au double entre tokens fréquents et rares ; c'est sur les rares que
les techniques de compression et de capacité se distinguent (études 08, 09, 11). Le modèle a appris la
forme d'un raisonnement, pas son enchaînement : dans l'exemple type, la première étape est juste, la
seconde repart de la donnée initiale au lieu du résultat intermédiaire. Faire « réfléchir » un modèle de
cette taille ajoute des occasions de se tromper.

**Limites.** 30 problèmes par case : une case vaut à ±8 points près. Les questions pièges ne sont pas notées automatiquement.

**Reproduire.** `python eval_detail.py <poids> [--ref <poids>]` ; `python eval_reasoning.py <poids> --langs zh,en`.
