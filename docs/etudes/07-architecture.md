# 07 — Forme du réseau : profondeur, largeur, partage des têtes, experts

**Question.** À nombre de paramètres voisin, une autre répartition fait-elle mieux que l'architecture d'origine (8 blocs de largeur 768, 8 têtes de requêtes pour 4 têtes clés/valeurs) ?

**Protocole.** 50M tokens, recette finale (Muon 0,01, batch 64), une graine par variante. Référence : 2,350.
Bruit : 0,012, donc seuls les écarts supérieurs à ~0,03 sont lisibles. Le temps est rapporté, car un
gain qui coûte du temps n'en est pas un à budget de calcul égal.

## Résultats

| Variante | Paramètres | val_loss | Écart | Temps | Verdict |
|---|---|---|---|---|---|
| Référence 8 × 768, 4 têtes K/V | 63,9M | 2,350 | — | — | — |
| 8 têtes K/V (aucun partage) | 68,6M | 2,336 | −0,014 | +8 % | dans le bruit, plus cher |
| Profond : 12 blocs × 640 | 66,0M | 2,343 | −0,007 | +12 % | dans le bruit, plus lent |
| Experts (MoE) : 4 FFN, 1 actif par token | 198M | 2,365 | +0,015 | **+48 %** | moins bon et bien plus lent |
| Large : 5 blocs × 960 | 64,2M | 2,366 | +0,016 | 0 % | légèrement moins bon |
| 1 tête K/V | 60,4M | 2,367 | +0,017 | −7 % | légèrement moins bon, plus rapide |

## Conclusion

Aucune variante ne bat nettement la référence. Le sens des écarts recoupe la littérature : la profondeur
vaut mieux que la largeur, et le partage des têtes coûte peu. L'échec du MoE s'explique par le budget
(chaque expert ne voit qu'un quart des tokens) et par une implémentation en boucle, peu efficace sur GPU.
En regard, l'optimisation avait fait gagner 0,60 : à cette échelle, la forme du réseau compte peu.

**Limites.** Une seule graine, donc aucun écart ici n'est établi. Budget court, défavorable aux variantes
qui ajoutent de la capacité (voir étude 06). Le learning rate, réglé pour la référence, l'avantage.

**Reproduire.** `trainer/sweeps/archi.txt`, `archi_suite.txt` ; options `--hidden_size`, `--num_hidden_layers`, `--num_key_value_heads`, `--use_moe`.
