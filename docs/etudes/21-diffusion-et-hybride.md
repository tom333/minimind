# 21 — Diffusion masquée et architecture hybride

**Question.** Peut-on générer du texte comme une image, en partant d'un brouillon qu'on affine (diffusion
masquée, MDLM / LLaDA), et combiner cette approche avec la génération mot à mot ?

**Principe.** Le « bruit » est le masquage : on masque une part aléatoire des tokens et le modèle, qui voit toute
la séquence, les retrouve. La génération part d'une séquence entièrement masquée et fixe à chaque tour les mots
les plus sûrs ; on peut ensuite re-masquer les moins sûrs et les prédire à nouveau (raffinage). L'architecture
hybride fait écrire un brouillon par la diffusion, puis le mode mot à mot du même modèle le relit, accepte les
mots plausibles et remplace le premier mot refusé (décodage spéculatif avec la diffusion comme brouillonneur).

## Résultats

Entraînement, 50M tokens (borne de diffusion : borne supérieure de la loss, défavorable à la diffusion) :

| Variante | Loss mot à mot | Borne de diffusion |
|---|---|---|
| Mot à mot seul (référence) | 3,462 | — |
| Diffusion depuis zéro | — | 4,689 |
| Diffusion à partir du modèle mot à mot complet | — | 4,147 |
| Mixte (un pas sur deux), depuis zéro | 3,838 | 4,371 |
| **Mixte, à partir du modèle complet, taux réduit, 100M tokens** | 3,023 (départ : 2,967) | **3,834** |

Génération sur 60 débuts de textes jamais vus, 32 mots (modèle mixte adapté) :

| Méthode | Acceptation (1er bloc) | Passes par mot | Loss jugée | Répétition | Temps |
|---|---|---|---|---|---|
| Mot à mot seul | — | 1,00 | 1,047 | 21 % | 13 s |
| Hybride, 2 tours par bloc | 61 % (16 %) | 1,16 | 1,143 | 23 % | 29 s |
| Hybride, 8 tours par bloc | 84 % (39 %) | 2,01 | 1,056 | 27 % | 40 s |

## Conclusion

Le mécanisme fonctionne, mais ne rapporte rien à cette taille : l'hybride achète de la qualité ou de la vitesse,
jamais les deux, et reste deux à trois fois plus lent en temps réel. Le blocage est le premier bloc, où le mot à
mot n'accepte que 16 à 39 % du brouillon : la diffusion reste trop loin du mot à mot (3,83 contre 3,02). Les
textes produits par les deux méthodes bouclent, ce qui limite la portée de la mesure de qualité. Apprendre les
deux modes coûte peu au mot à mot (+0,056) avec un taux d'apprentissage réduit. À rejuger avec plus
d'entraînement en diffusion et un modèle plus grand ; la mesure est prête.

**Limites.** Une graine ; la loss jugée est flattée par les répétitions ; génération sans cache.

**Reproduire.** `--objective mdm|both` ; `trainer/diffusion.py` (`--block`, `--refine`, `--hybride`),
`trainer/eval_hybride.py` ; études `diffusion.yaml`, `hybride.yaml`.
