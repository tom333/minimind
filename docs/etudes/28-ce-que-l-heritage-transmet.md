# 28 — Ce que l'héritage transmet, et ce qu'il perd

**Question.** Où est la valeur d'un modèle hérité, et pourquoi l'enfant ne raisonne-t-il pas comme son parent ?

**Protocole.** Enfant de 150M taillé dans Qwen3-0.6B (8 blocs sur 28). Dissection : on abîme une propriété à la fois
avant 10M tokens d'entraînement. Puis bancs standards de l'enfant de 4 h et du parent, et calcul d'école (GSM8K,
250 problèmes) pour trois modèles publics.

## Résultats

| Ce qui vient du parent | Part du bénéfice de l'héritage |
|---|---|
| Tout | 100 % |
| Embedding + attention | 58 % |
| Signes seuls des poids (−a, 0, +a) | au moins 40 % |
| Embedding seul | 26 % |
| Blocs sans leur embedding | 2 % |
| Blocs dans l'ordre inverse ; échelle seule | aucun |

| Épreuve (%) | Hasard | Enfant 150M, 4 h | Parent Qwen3-0.6B |
|---|---|---|---|
| Grammaire française | 25 | 69,7 | 68,1 |
| Vocabulaire français | 25 | 55,5 | 59,7 |
| Compréhension d'un texte en français | 25 | 23,6 | 56,6 |
| Sciences (SciQ) | 25 | 65,5 | 91,3 |
| Calcul d'école (GSM8K) | ~0 | — (contexte trop court) | 53,2 |

| Calcul d'école (GSM8K) | SmolLM2-135M | SmolLM2-360M | Qwen3-0.6B |
|---|---|---|---|
| % résolus | 1,2 | 4,8 | 53,2 |

À 10M tokens : 12 blocs distincts au lieu de 8 font gagner −0,08 à −0,13 bit/octet et 6 points en sciences ; 4 blocs
exécutés deux fois, −0,008.

## Conclusion

La valeur héritée tient en un système de coordonnées (l'embedding) et un câblage écrit dedans, surtout l'attention.
L'enfant garde la langue du parent mais perd sa compréhension, ses connaissances et son calcul : ce que la coupe en
profondeur (8 blocs sur 28), nos données (sans mathématiques) et le contexte de 340 tokens ont jeté. Le raisonnement
existe à 600M (53 % au calcul d'école) mais pas à 360M sans données de raisonnement massives (4,8 %) : les données
comptent au moins autant que la taille. Rendre de la profondeur par boucle ne marche pas à cette échelle ; rendre de
vrais blocs, si.

**Limites.** Une graine ; XNLI, SciQ et GSM8K sur 1 000 ou 250 questions. 96 % des questions de compréhension
tiennent dans le contexte de l'enfant : son échec n'est pas un effet de troncature.

**Reproduire.** `trainer/heritage_dissection.py`, études `heritage_dissection.yaml`, `heritage_hybrides.yaml`,
`heritage_boucle.yaml` ; `trainer/eval_bench.py` (modèles hérités : `poids.pth@fiche.json`).
