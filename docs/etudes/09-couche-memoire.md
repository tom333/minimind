# 09 — Couche mémoire à clés produit

**Question.** Peut-on ajouter beaucoup de paramètres pour presque aucun calcul, avec une table que chaque token consulte ? (Lample et al., 2019 ; Berges et al., 2024, « Memory Layers at Scale »)

**Principe.** Une table de 65 536 lignes (+50,9M de paramètres) placée au milieu du réseau. Chaque token
calcule une requête, retrouve les 8 lignes les plus proches par « clés produit » (deux fois 256 sous-clés
au lieu de 65 536 comparaisons) et n'en lit que celles-là.

**Protocole.** Même recette et même graine que la référence, à deux budgets : 50M tokens, puis une epoch
complète (332M). Bruit : 0,012. Plus bas = mieux.

## Résultats

| Mesure | Sans mémoire | Avec mémoire | Écart | Verdict |
|---|---|---|---|---|
| val_loss à 50M tokens | 2,338 | 2,338 | 0,000 | aucun effet |
| val_loss à 332M tokens | 2,010 | 2,001 | −0,010 | dans le bruit |
| Loss sur les tokens rares (332M) | 2,804 | 2,798 | −0,006 | aucun effet là où on l'attendait |
| Durée du run | 145 min | 154 min | +6 % | bon |
| Paramètres | 63,9M | 114,8M | +80 % | cher pour ce gain |

Diagnostic du modèle à 50M tokens (learning rate de la table multiplié par 10) :

| Mesure | Valeur | Lecture |
|---|---|---|
| Amplitude de la sortie de la mémoire | 7,1 contre 12 pour le FFN | la table pèse dans le calcul |
| Lignes consultées | 13 400 sur 65 536 | la table est réellement utilisée |
| Loss si on débranche la mémoire | +0,057 | le modèle s'appuie dessus |

## Conclusion

La couche fonctionne et tient sa promesse de coût, mais elle ne rapporte rien de mesurable. Le diagnostic
est instructif : le modèle a déplacé une partie du travail du FFN vers la table, sans rien apprendre de
plus. À 50M tokens, c'est attendu (limite de données, étude 06). À 332M tokens, une avance régulière de
0,010 à 0,017 apparaît pendant le run, mais reste de la taille du bruit.

**Limites.** Un seul run par budget. Les gains publiés sont obtenus après mille milliards de tokens, avec
des tables mille fois plus grandes.

**Reproduire.** `--memory_layers 4 --memory_lr_ratio 10` ; études `memoire.yaml`, `memoire_full.yaml` ; `model/memory.py`.
