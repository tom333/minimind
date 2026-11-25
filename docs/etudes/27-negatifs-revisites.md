# 27 — Résultats négatifs, testés autrement

**Question.** Les idées qui avaient échoué (distillation, comptage, ternaire, régression, hybride) marchent-elles si
on lève la cause de leur échec ? Et deux idées nouvelles du 25 novembre : la trajectoire des poids et l'embedding
calculé par comptage.

**Protocole.** Pour chaque idée, la cause supposée de l'échec est retirée, le reste de la recette ne change pas.
Repères de la même étude ou d'une étude au protocole identique. Plus bas = mieux.

## Résultats

| Idée | Ce qui change | Mesure | Repère | Verdict |
|---|---|---|---|---|
| Distillation | probabilités de l'enseignant calculées une fois (12 min), relues comme les tokens | 3,362 en 32 min | témoin 3,456 en 28 min ; en ligne 3,364 en 42 min | **gain**, si les probabilités servent plusieurs fois |
| Comptage | loss de mélange pendant l'entraînement, occurrence propre retirée des comptes | modèle + comptage 3,476 ; modèle seul 5,474 | témoin + comptage 3,209 | négatif : le modèle se repose sur le comptage |
| Ternaire hérité | 16 bits d'abord, bascule en ternaire à mi-parcours ou aux 4/5 | 3,815 ; 5,802 | 16 bits hérité 3,257 ; ternaire dès le départ 3,930 | gain faible ; bascule tardive : effondrement |
| Régression → largeur | Qwen3-1.7B réduit de 2 048 à 1 024 dimensions (rotation vers les directions principales) | 3,403 | enfant de Qwen3-0.6B 3,289 ; zéro 4,103 | **marche en partie** : 83 % du bénéfice |
| Hybride diffusion | rejugé sur l'enfant hérité de 4 h | 9 à 15 % de brouillons acceptés au premier bloc ; 1,35 à 2,79 passes par mot | à 71M : 16 à 39 % | négatif ; l'adaptation coûte +0,30 en mot à mot |
| Trajectoire des poids | sauter la fin de l'entraînement, ou moyenner des copies | saut : toujours pire ; moyenne : −0,005 | fin réelle 3,420 | négatif : les poids zigzaguent |
| Embedding par comptage | vecteurs des tokens calculés (cooccurrences + SVD), 5 min de processeur | −0,082 à 50M tokens ; −0,033 à 200M | 3,462 ; 3,120 | **gain** qui s'use avec la durée |
| Boucle (Loop Transformer) | 4 blocs sur 8 exécutés deux fois, portes par passage | −0,008 bit/octet, +35 % de temps | 12 blocs distincts : −0,08 à −0,13 | négatif à cette échelle |

Loss en tokens Qwen pour les études d'héritage, sur notre vocabulaire pour les autres.

## Conclusion

Deux idées se rachètent : la distillation, quand l'enseignant ne passe qu'une fois, et la réduction de largeur, qui
ouvre l'héritage depuis des parents plus gros. L'embedding par comptage est un gain réel mais qui s'use : utile
pour les essais courts. Les autres confirment leur échec, pour des raisons maintenant identifiées.

**Limites.** Une graine par essai ; budgets courts (10 à 50M tokens).

**Reproduire.** Études `distillation_hors_ligne*.yaml`, `comptage_entrainement.yaml`, `heritage_ternaire_fin.yaml`,
`heritage_largeur.yaml`, `heritage_hybride.yaml`, `trajectoire.yaml`, `embedding_comptage*.yaml`, `heritage_boucle.yaml`.
