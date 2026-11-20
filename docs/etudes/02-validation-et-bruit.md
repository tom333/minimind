# 02 — Peut-on se fier à la mesure ? Jeu de validation et bruit

**Question.** La loss de validation mesure-t-elle ce qu'on croit, et à partir de quel écart une différence entre deux runs est-elle réelle ?

**Protocole.** Inspection du jeu de validation initial (les 2 000 derniers documents du fichier), comparaison
à un tirage aléatoire de 2 000 documents, calcul de repères simples, puis répétition du même entraînement
avec deux graines aléatoires.

## Résultats

Le fichier de corpus est un assemblage de sources rangées bout à bout :

| Zone du fichier | Part de caractères ASCII | Contenu |
|---|---|---|
| Début | 14 % | instructions courtes en chinois |
| Milieu | 16 % | réponses longues en chinois |
| 2 000 derniers documents | **100 %** | questions à choix multiples en anglais |

Le premier jeu de validation ne mesurait donc que de l'anglais, soit environ 0,2 % du corpus.

Repères sur le jeu de validation corrigé (tirage aléatoire, graine fixe) :

| Modèle | val_loss | Lecture |
|---|---|---|
| Hasard uniforme | 8,76 | ln(6 400) |
| Fréquences des tokens seules | 7,40 | sait quels tokens sont courants |
| Modèles du premier balayage, réévalués | 3,10 | contre 3,37 affichés sur l'ancien jeu |

Bruit entre deux graines, à réglages identiques (50M tokens) :

| Recette | Graine 42 | Graine 2 | Écart |
|---|---|---|---|
| Réglages d'origine | 2,950 | 3,005 | 0,055 |
| Recette finale (Muon, batch 64) | 2,350 | 2,338 | 0,012 |

## Conclusion

Deux règles en découlent pour toutes les études suivantes. La validation est tirée au hasard, avec une
graine fixe. Un écart inférieur à 0,012 de val_loss n'est pas interprété ; au-delà de trois fois cette
valeur, il est considéré comme réel. Plusieurs « gains » des premiers balayages étaient dans le bruit.

**Limites.** Le bruit est estimé sur deux graines seulement. La loss moyenne reste dominée par les tokens
fréquents, d'où l'évaluation détaillée de l'étude 13.

**Reproduire.** Tirage dans `trainer/train_pretrain.py` (`random.Random(0).sample`) ; analyse du fichier : `scripts/analyze_data.py`.
