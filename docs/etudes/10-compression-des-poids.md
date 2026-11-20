# 10 — Compresser les poids : ce qui est redondant et ce qui ne l'est pas

**Question.** Les poids d'un modèle entraîné contiennent-ils de la redondance exploitable : par compression de fichier, par partage des valeurs identiques, par voisinage, par neurones inutiles ?

**Protocole.** Mesures directes sur un modèle entraîné (point de contrôle à mi-parcours du run complet,
val_loss 2,338), sans réentraînement. Poids en 16 bits (bf16).

## Résultats

Compression sans perte, comme un fichier :

| Méthode | Taille obtenue | Lecture |
|---|---|---|
| zip | 79 % | faible |
| xz | 71 % | faible |
| Limite théorique (entropie) | ~66 % | le signe porte 1,00 bit sur 1, l'exposant 2,64 sur 8, la mantisse 6,96 sur 7 |

Valeurs partagées :

| Mesure | Résultat |
|---|---|
| Valeurs distinctes parmi 63,9M de poids | 4 289 : tous les poids ont un jumeau exact |
| Coût d'une référence vers l'une de ces valeurs | 12,1 bits, contre 16 pour le poids lui-même |

Table de valeurs imposée par matrice (regroupement k-means, sans réentraînement), plus bas = mieux :

| Valeurs par matrice | Bits par poids | Taille | val_loss | Verdict |
|---|---|---|---|---|
| aucune table | 16 | 128 Mo | 2,338 | référence |
| 256 | 8 | 64 Mo | 2,347 | perte dans le bruit |
| 16 | 4 | 32 Mo | 2,395 | perte modeste (+0,057) |
| 4 | 2 | 16 Mo | 2,961 | effondrement (+0,62) |

Structure exploitable ou non (une matrice du FFN, 2 432 × 768) :

| Hypothèse | Mesure | Verdict |
|---|---|---|
| Deux poids voisins se ressemblent | corrélation 0,000 (photo : ~0,95) | faux : l'ordre des neurones est arbitraire |
| Une table par bloc de 64 × 64 aide | erreur 0,0119, identique à la table globale | faux |
| Une table par ligne (par neurone) aide | erreur 0,0103, soit −13 % | vrai : l'amplitude varie d'un facteur 244 entre neurones |
| Des neurones sont inutilisés | 19 sur 19 456 (0,1 %) | faux |
| Les matrices sont de rang faible | 49 à 72 % des directions pour 90 % de l'énergie | faux |

## Conclusion

Le modèle utilise presque toute sa capacité : pas de neurones morts, pas de rang faible, pas de voisinage.
La seule redondance est l'ordre de grandeur des poids, d'où le gain limité à 30 % de la compression sans
perte. La compression avec perte est en revanche efficace : deux fois moins de place pour une perte
indétectable, quatre fois moins pour +0,06. La structure pertinente est la ligne, c'est-à-dire le neurone.

**Limites.** Mesures sur un seul point de contrôle et, pour la structure, sur une seule matrice.

**Reproduire.** Mesures faites par scripts ponctuels pendant l'analyse ; elles sont à reprendre dans un script dédié.
