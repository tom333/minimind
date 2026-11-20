# 01 — Débit d'entraînement : où part le temps de calcul ?

**Question.** Le script d'origine exploite-t-il bien le GPU, et sinon, que peut-on récupérer sans toucher au modèle ?

**Protocole.** Modèle de référence (63,9M de paramètres), RTX 3060 12 Go, corpus `pretrain_t2t_mini` (1,27M de documents).
Mesure du débit en tokens par seconde sur 600 pas, puis profilage d'un pas d'entraînement
(`trainer/profile_step.py`) : temps par phase et temps GPU par famille de noyaux.

## Résultats

| Mesure | Valeur | Repère | Verdict |
|---|---|---|---|
| Tokens utiles par séquence (script d'origine) | 160 sur 340 (47 %) | 100 % = aucun gaspillage | mauvais : 53 % du calcul porte sur du remplissage |
| Texte perdu par la troncature à 340 tokens | ~40 % du corpus | 0 % | mauvais : la longueur moyenne (261) dépasse largement la médiane (141) |
| Débit utile, script d'origine | ~13 000 tokens/s | — | référence |
| + regroupement des documents en blocs pleins (*packing*) | ~28 000 tokens/s | ×2,1 | bon |
| + `torch.compile` | ~40 000 tokens/s | ×1,4, loss identique (6,482 contre 6,487) | bon |
| + batch réel de 64 au lieu de 32 × 2 accumulés | 42 500 tokens/s | +6,5 %, calcul mathématiquement identique | bon |
| Tokens appris par epoch | 332M contre 203M | — | +63 % de données en 25 % de temps en moins |

Profil d'un pas (configuration finale, optimiseur Muon) :

| Phase | Part du temps | | Famille de noyaux GPU | Part |
|---|---|---|---|---|
| Rétropropagation | 57 % | | Multiplications de matrices | 78 % |
| Passe avant | 28 % | | Opérations fusionnées (normes, SiLU, RoPE) | 13 % |
| Optimiseur | 15 % | | Attention | 6 % |
| Chargement des données | 0,1 % | | Reste | 3 % |

Essai d'une librairie de noyaux spécialisés (Liger Kernel 0.8.5, `trainer/bench_liger.py`) :

| Configuration | Débit | Verdict |
|---|---|---|
| PyTorch seul | 27 800 tokens/s | — |
| Liger seul | 38 400 tokens/s | +38 % |
| `torch.compile` seul | 44 600 tokens/s | la meilleure |
| `torch.compile` + Liger | 42 900 tokens/s | −4 % ; plantage si les noyaux Liger sont compilés avec le reste |

## Conclusion

Le débit utile a été multiplié par trois sans modifier le modèle. Le gain vient de ne plus calculer de
vide, pas d'un calcul plus rapide : le GPU tournait déjà à pleine charge. Après ces corrections, 78 % du
temps passe dans des multiplications de matrices déjà optimales : il ne reste que quelques pourcents à
gagner côté système. Liger n'apporte rien par-dessus `torch.compile`, qui fusionne déjà les mêmes opérations.

**Limites.** Mesures sur une seule carte. Le regroupement en blocs laisse un document « voir » la fin du
précédent ; l'effet n'a pas été mesuré isolément.

**Reproduire.** `--packing 1 --use_compile 1 --batch_size 64` dans `trainer/train_pretrain.py` ; commits `460bb81`, `2462132`, `4257f94`.
