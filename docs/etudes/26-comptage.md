# 26 — Que remplace le comptage ?

**Question.** Les longues heures d'entraînement apprennent-elles autre chose que ce qu'une passe de comptage
sur le corpus sait déjà ? (kNN-LM, Khandelwal et al. 2020 ; ∞-gram, Liu et al. 2024)

**Principe.** Un prédicteur par pur comptage : fréquences des suites de 1 à 10 tokens dans tout le corpus
d'entraînement, lissées par ordre décroissant. Quelques minutes de processeur, aucune carte graphique. On le
mélange à un modèle : μ·comptage + (1−μ)·modèle, μ ne dépendant que du contexte.

**Protocole.** A = modèle entraîné peu, B = même modèle entraîné longtemps. Les réglages du comptage et du
mélange sont faits sur une moitié des blocs de validation, les mesures sur l'autre. Plus bas = mieux.

## Résultats

Où part le temps (71M, même recette) : 27 min → 3,46 ; 96 min → 3,12 ; 8 h → 2,97. Les 6 h 30 finales gagnent 0,15.

| Loss | Comptage seul | A | B | A + comptage | B + comptage |
|---|---|---|---|---|---|
| 71M, corpus web courant (A : 27 min, B : 8 h) | 3,986 | 3,444 | 2,950 | 3,215 | 2,849 |
| 71M, corpus sélectionné (A : 27 min, B : 1 Md de tokens) | 3,883 | 3,369 | 2,885 | 3,125 | 2,763 |
| Enfant hérité, tokens Qwen (A : 29 min, B : 58 min) | 3,738 | 2,966 | 2,833 | 2,776 | 2,680 |

Par famille de tokens (corpus web courant), selon la plus longue suite précédente déjà vue dans le corpus :

| Contexte déjà vu | Part des tokens | Comptage seul | A | B | B + comptage |
|---|---|---|---|---|---|
| 2 à 5 tokens | 75,5 % | 4,29 à 4,47 | 3,51 à 3,62 | 3,02 à 3,12 | 3,01 à 3,09 |
| 6 à 8 tokens | 14,2 % | 3,88 | 3,44 | 2,94 | 2,87 |
| 9 tokens ou plus | 9,3 % | 1,16 | 2,25 | 1,73 | 0,93 |

En bits par octet sur la validation du corpus sélectionné (mêmes 2 000 documents pour tous) :

| Modèle | Sans comptage | Avec comptage |
|---|---|---|
| Notre 71M parti de zéro, 27 min | 1,254 | 1,163 |
| Notre 71M, 1 Md de tokens (~8 h) | 1,074 | 1,029 |
| Hérité 150M, 29 min | 1,144 | 1,071 |
| Hérité 150M, 58 min | 1,092 | 1,033 |

## Conclusion

- **Tout ce que sait le comptage, l'entraînement l'apprend en 9 minutes** (3,99). Remplir un modèle à partir des
  seuls comptages plafonne à ce niveau.
- **Les longues heures n'apprennent pas des suites rares** : elles gagnent environ 0,49 dans toutes les familles
  de tokens, y compris là où rien n'est retrouvable. Elles affinent la fonction générale du modèle.
- **Le comptage est un complément gratuit** : −0,045 bit/octet sur notre modèle de 8 h, plus que le doublement du
  modèle. Les trois quarts de ce gain viennent des passages déjà présents mot pour mot dans le corpus.
- **Héritage et comptage s'additionnent** : une heure d'entraînement hérité avec comptage (1,033) égale nos 8 h
  avec comptage (1,029).

**Limites.** Lissage rudimentaire. Mesuré sur la validation du corpus (9 à 14 % de tokens en passages répétés),
pas sur les jeux de test. Le résultat est un modèle accompagné d'une table, pas un modèle seul ; la génération
avec la table n'est pas codée.

**Reproduire.** `trainer/diag_comptage.py` (modèle hérité : `poids.pth@fiche.json`).
