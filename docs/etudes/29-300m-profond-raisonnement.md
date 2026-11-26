# 29 — Un 300M profond qui apprend à calculer, mais seulement en anglais

**Question.** L'enfant à 8 blocs a perdu le raisonnement de son parent. En gardant plus de profondeur, en allongeant le
contexte et en ajoutant des problèmes résolus aux données, peut-on le retrouver ? Et passe-t-il en français ?

**Protocole.** Qwen3-0.6B taillé à 16 blocs sur 28 (276M paramètres ; avant entraînement, loss 5,03 contre 9,40 à
8 blocs). Contexte de 1 024 tokens. 150M tokens d'un mélange : corpus sélectionné et 395 000 problèmes de
mathématiques résolus pas à pas, en anglais (MetaMathQA, MIT ; 24 % du texte). Taux réduit, 5 h 20 sur la RTX 3060.
Épreuves : GSM8K (250 problèmes, 5 exemples dans la consigne) ; MGSM (les mêmes problèmes en anglais et traduits en
français, sans exemple) ; bancs français ; bits par octet sur nos cinq jeux de test.

## Résultats

| Mesure | 300M profond | Enfant 8 blocs (4 h) | Parent Qwen3-0.6B | SmolLM2-360M |
|---|---|---|---|---|
| Calcul d'école GSM8K (±2,7) | **24,8 %** | impossible (contexte) | 53,2 % | 4,8 % |
| Même épreuve, en anglais / en français (MGSM) | 20,4 % / **2,0 %** | — | 34,8 % / 26,4 % | — |
| Compréhension d'un texte FR (hasard 25 %) | 25,0 % | 23,6 % | 56,6 % | 24,6 % |
| Sciences (SciQ) | 66,9 % | 65,5 % | 91,3 % | 86,5 % |
| Grammaire / vocabulaire FR | 68,1 / 56,3 % | 69,7 / 55,5 % | 68,1 / 59,7 % | 44,5 / 37,0 % |
| Bits par octet, web FR sélectionné / anglais | 1,034 / 1,023 | 1,031 / 1,034 | 1,009 / 0,823 | 1,292 / 0,792 |

Ce que le modèle écrit sur un même problème : en anglais, il pose les étapes et calcule (« 3 × 60 = 180 … 180 × 3 =
540 ») ; en français, il recopie une phrase de l'énoncé en boucle et ne calcule jamais.

## Conclusion

À 276M paramètres, le calcul d'école s'apprend : 5 fois mieux que SmolLM2-360M, la moitié du parent. Mais la
compétence reste attachée à la langue des données : 2 % en français, contre 76 % de transfert chez le parent. Elle ne
s'étend pas non plus à la lecture : la compréhension d'un texte reste au hasard.

**Limites.** Les problèmes d'entraînement sont des réécritures de ceux d'entraînement de GSM8K : le score mesure
surtout un apprentissage ciblé. Une graine ; trois changements à la fois (profondeur, contexte, données), dont la part
de chacun n'est pas isolée.

**Reproduire.** `python -m corpus.melange_raisonnement` ; `heritage.py tailler --blocs 16 --liste
0,1,2,3,4,5,16,17,18,20,21,22,23,25,26,27` ; étude `raisonnement_300m.yaml` ; `eval_bench.py` (tâches `gsm8k`,
`mgsm_native_cot_en`, `mgsm_native_cot_fr`).
