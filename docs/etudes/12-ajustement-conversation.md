# 12 — Du modèle de langue à l'assistant : l'ajustement sur conversations (SFT)

**Question.** Quels réglages pour transformer le modèle pré-entraîné en assistant, et le résultat vaut-il le modèle officiel du projet ?

**Protocole.** Départ : modèle du run complet (étude 06). Données : 906 000 conversations
(`sft_t2t_mini`), 2 000 mises de côté pour la validation. La loss ne porte que sur les réponses de
l'assistant. Étude courte de 3 000 pas pour choisir optimiseur et learning rate, puis une epoch complète
en conservant le meilleur modèle en validation. Plus bas = mieux.

## Résultats

Les données :

| Mesure | Valeur | Conséquence |
|---|---|---|
| Longueur médiane d'une conversation | 491 tokens (plafond 768) | 37 % de remplissage si tout est complété à 768 |
| Tokens par epoch | 440M | plus que le pré-entraînement (332M) |
| Conversations avec raisonnement / avec outils | 30 % / 9 % | — |

Les conversations de longueurs voisines sont regroupées et chaque batch est coupé à sa plus longue
conversation, ce qui supprime l'essentiel du remplissage. Epoch mesurée : 4h32, à 26 000 tokens/s
(moins que les 40 000 du pré-entraînement : batches plus petits, longueurs variables).

Choix du réglage (3 000 pas, départ à ~2,0) :

| Réglage | val_loss | Évolution | Verdict |
|---|---|---|---|
| AdamW 1e-5 (réglage d'origine) | 1,675 | à plat dès la moitié | trop lent |
| AdamW 1e-4 | 1,528 | ralentit en fin de course | bon |
| **Muon 0,002 + AdamW 1e-4** | **1,513** | encore en nette baisse | retenu |
| Muon 0,005 + AdamW 3e-4 | 1,595 | démarre à 2,04, au-dessus du point de départ | abîme d'abord le modèle pré-entraîné |

Epoch complète (4h32) :

| Modèle | val_loss sur les conversations | Lecture |
|---|---|---|
| Pré-entraînement seul | 2,380 | point de départ |
| **Après ajustement** | **1,262** | baisse à chacune des 29 mesures, sans remontée |
| Modèle officiel `minimind-3` | 1,180 | a vu ces conversations à l'entraînement, et davantage de données |

Sur sept questions posées aux deux modèles, les réponses sont de niveau comparable : le nôtre est souvent
plus propre sur la forme, l'officiel plus fiable sur les faits. Les deux inventent hors des évidences et
aucun n'enchaîne deux opérations (étude 13).

## Conclusion

Le réglage d'origine (1e-5) est nettement trop bas. Muon reste le meilleur en ajustement, à condition de
diviser son learning rate par cinq par rapport au pré-entraînement. Aucun surapprentissage sur une epoch.
L'écart de 0,08 avec le modèle officiel est faible au regard de son avantage : il a été entraîné sur les
conversations qui nous servent de validation.

**Limites.** Une seule graine. La comparaison des réponses est une lecture humaine sur sept questions.

**Reproduire.** `--task sft` ; études `sft_lr.yaml`, `sft_full.yaml` ; commit `1a835c4`.
