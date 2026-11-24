# 25 — Héritage : tailler un 150M dans un modèle existant

**Question.** Partir des poids d'un modèle déjà entraîné, au lieu du hasard, fait-il gagner du temps à qualité
égale ? (Sheared-LLaMA, Xia et al. 2023 ; Minitron, Muralidharan et al. 2024)

**Principe.** Notre code de modèle a la même architecture que Qwen3. Le parent est Qwen3-0.6B-Base (596M
paramètres, 28 blocs de largeur 1 024, licence Apache 2.0). Deux coupes, puis un entraînement ordinaire :
1. *Vocabulaire* : 151 669 → 24 000 tokens, ceux qui servent dans notre corpus. Les fusions BPE gardées le sont
   avec leurs ancêtres : un mot dont les tokens sont gardés se découpe comme chez le parent.
2. *Blocs* : 8 gardés sur 28, en retirant à chaque tour celui dont l'absence gêne le moins.

**Protocole.** Enfant de 150,4M. Témoin : la même forme partie de poids aléatoires, mêmes données (corpus
sélectionné), même vocabulaire. Le tokenizer n'étant pas le nôtre, la comparaison avec nos autres modèles se
fait en bits par octet sur les cinq jeux de test. Plus bas = mieux.

## Résultats

| Contrôle | Mesure | Repère |
|---|---|---|
| Occurrences couvertes par les 24 000 tokens | 97,5 % | textes 3,1 % plus longs en tokens |
| Écart de nos sorties avec le code d'origine du parent | 0,00008 | attendu sous 0,01 |
| Parent entier, vocabulaire réduit | loss 2,84 | hasard 10,09 |
| Enfant à 8 blocs, avant entraînement | 9,40 | 13,6 et 14,6 pour deux autres choix de blocs |

| Essai (30M tokens, 45 min tout compris) | val_loss (tokens Qwen) |
|---|---|
| Parti de zéro | 3,396 |
| Hérité, notre taux d'apprentissage | 3,056 |
| **Hérité, taux réduit à 30 %** | **2,982** |

| Bits par octet | Hérité 29 min | Hérité 58 min | Hérité 2 h | Hérité 4 h | Zéro, même forme, 29 min | Notre 71M zéro, 27 min | Notre 71M, 1 Md de tokens (~8 h) |
|---|---|---|---|---|---|---|---|
| Anglais | 1,123 | 1,085 | 1,055 | **1,034** | 1,347 | 1,311 | 1,120 |
| Web français sélectionné | 1,154 | 1,103 | 1,061 | **1,031** | 1,284 | 1,232 | 1,070 |
| Web français courant | 1,290 | 1,238 | 1,193 | **1,160** | 1,426 | 1,373 | 1,203 |
| Wikipédia | 0,685 | 0,597 | 0,531 | **0,488** | 1,082 | 0,976 | 0,583 |
| Wikisource | 1,296 | 1,241 | 1,196 | **1,161** | 1,443 | 1,388 | 1,212 |

Remplir l'enfant par régression sur les états internes du parent, sans entraînement de bout en bout (chaque bloc
de l'enfant remplace une tranche de 3 ou 4 blocs du parent) :

| Façon de remplir | loss | Temps de carte |
|---|---|---|
| Premier bloc de chaque tranche, tel quel | 14,28 | 0 |
| Moindres carrés sur les deux matrices de sortie de chaque bloc | 6,61 | 9,9 min |
| Chaque bloc entraîné seul à imiter la sortie de sa tranche | 8,59 | 6,2 min |
| Repère : entraînement ordinaire de l'enfant taillé | 3,48 puis 3,28 | 5,3 puis 10,6 min |

## Conclusion

- **L'héritage fonctionne.** À temps égal, l'enfant bat notre meilleur départ de zéro de 0,08 à 0,29 bit/octet.
  En 2 h, il passe devant notre 71M de 8 h sur les cinq jeux : environ 4 fois moins de temps à qualité égale.
  En 4 h, il le devance de 0,04 à 0,10. Le gain par doublement de durée diminue : 0,05, puis 0,04, puis 0,03.
- **La coupe détruit le modèle, l'entraînement le répare vite** : de 11 à 4,2 de loss en une minute.
- **Il faut réduire le taux d'apprentissage** : le nôtre efface une partie de l'héritage.
- **Imiter les états internes du parent perd nettement** contre l'entraînement ordinaire : un bloc ne peut pas
  refaire le travail de quatre ; l'enfant doit trouver son propre chemin.
- **Le parent limite le résultat** : Qwen3-0.6B est fort en anglais (0,823) mais à peine meilleur que notre 71M
  en français (1,009 contre 1,070 sur le web sélectionné).

**Limites.** Une graine. L'enfant (150M) est comparé à un 71M : nous n'avons pas de 150M parti de zéro entraîné
longtemps. Le découpage français/anglais de la validation n'est pas fiable dans ces études (même token pour le
début et la fin des documents).

**Reproduire.** `trainer/heritage.py vocabulaire` puis `tailler` ; études `heritage.yaml`, `heritage_long.yaml`,
`heritage_2h.yaml`, `heritage_4h.yaml` ; `trainer/heritage_regression.py`.
