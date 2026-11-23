# 18 — Mixture-of-Depths : router les tokens, et un défaut de compilation

**Question.** Ne faire traverser un bloc sur deux qu'à une partie des tokens (Raposo et al., DeepMind 2024)
accélère-t-il l'entraînement sans trop dégrader le modèle ?

**Principe.** Dans les blocs impairs, un routeur (une projection) note chaque token ; à l'entraînement, les k
mieux notés traversent le bloc, les autres le contournent. La sortie du bloc est multipliée par la note. À
l'évaluation, la décision doit être causale : une perte auxiliaire apprend au routeur à ce que « note > 0 »
prédise le choix. Les deux chemins de calcul donnent le même résultat à 4e-6 près (test automatique).

## Résultats

| Capacité | val_loss (50M tokens) | Débit (banc) | Mémoire |
|---|---|---|---|
| Sans routage | 3,462 | 36 300 tokens/s | 8,2 Go |
| 50 % | 3,516 | +18 % | 7,2 Go |
| 25 % | 3,529 | +33 à +41 % | 6,6 Go |
| 12,5 % | 3,538 | +41 à +51 % | 6,3 Go |

À temps égal (fiche 17), la capacité 25 % finit à 3,151 contre 3,120. À 150M paramètres, elle fait tenir le
modèle en mémoire avec un lot de 64 et réduit le temps par token d'environ 30 % (fiche 19).

**Le défaut.** Le premier run à 25 % de chaque étude perdait 15 minutes avant son premier pas. Mesure du temps
de compilation du premier pas selon le nombre de tokens retenus (RTX 3060 ; T4 de Colab) :

| Tokens retenus | Avec tri | Sans tri |
|---|---|---|
| sans routage | 22 s | 22 s |
| 42 | 46 s | 27 s |
| 85 | **205 s** (1 058 s sur T4) | **27 s** |
| 170 | 28 s | 27 s |

Cause : `torch.compile` transforme un petit tri (`sort`, ici pour remettre les tokens retenus dans l'ordre du
texte) en tri bitonique entièrement déroulé, dont la compilation explose pour certaines tailles. Aucune issue
publique ne décrit ce cas précis. Contournement : un second `topk` sur « masque × rang inversé » donne les
mêmes indices sans tri (vérifié en fp32, bf16 et fp16, pour 340 à 2 048 tokens).

## Conclusion

La technique tient sa promesse de vitesse et de mémoire, pour 0,05 à 0,08 de loss en plus à budget égal. Elle
ne gagne pas à temps égal sur le 71M, mais devient utile dès que la mémoire contraint (150M). Le défaut de
compilation est un piège général : tout petit tri dans un graphe compilé est suspect.

**Limites.** Une graine ; séquences de 340 tokens, alors que le papier vise 2 048 (capacité optimale différente).
La génération avec cache n'est pas gérée pour ces modèles.

**Reproduire.** `--mod_capacity 0.25` ; `trainer/diag_compile_mod.py` ; études `mod.yaml`, `mod_iso2.yaml`, `regime.yaml`.
