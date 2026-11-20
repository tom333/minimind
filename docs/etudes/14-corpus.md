# 14 — Les données : qualité du corpus d'origine, construction du corpus français / anglais

**Question.** Le corpus d'origine est-il propre, et comment construire un corpus bilingue traçable ?

**Protocole.** Analyse du corpus chinois (`scripts/analyze_data.py`) : doublons exacts, quasi-doublons par
MinHash sur un échantillon de 100 000 documents, structure du fichier. Puis construction du corpus
français / anglais en assets Dagster : lecture en flux, nettoyage, déduplication, mélange à ratio paramétrable.

## Résultats

Corpus d'origine (1,27M de documents) :

| Mesure | Valeur | Verdict |
|---|---|---|
| Doublons exacts | 0,02 % | très propre |
| Quasi-doublons (similarité ≥ 0,8) | 0,09 % de l'échantillon | très propre |
| Documents de moins de 50 caractères | 0,44 % | négligeable |
| Licence | Apache 2.0 et CC BY-NC 2.0 | une partie en usage non commercial |

Corpus français / anglais :

| Source | Langue | Documents | Texte | Licence | Filtre |
|---|---|---|---|---|---|
| Wikipédia (20231101) | français | 195 237 | 1,26 Go | CC BY-SA 3.0 | sections de références retirées |
| FineWeb-2 | français | 544 065 | 1,59 Go | ODC-By | score de langue ≥ 0,9 |
| FineWeb-Edu (échantillon 10BT) | anglais | 320 739 | 1,43 Go | ODC-By | score éducatif ≥ 3, soit 45 % des pages gardées |

Mélange retenu : 1 008 282 documents, 3,99 Go, 64,3 % de français, soit 1,02 milliard de tokens.

Tokenizer commun (BPE, 16 000 tokens), mesuré sur des documents non vus :

| Mesure | Français | Anglais | Ancien tokenizer |
|---|---|---|---|
| Caractères par token (plus haut = mieux) | 3,74 | 3,94 | — |
| Tokens par mot (plus bas = mieux) | 1,67 | 1,55 | — |
| Phrase de test française | 14 tokens | — | 27 tokens |

## Conclusion

Le corpus d'origine n'a rien à gagner à un nettoyage supplémentaire. Le corpus bilingue est trois fois plus
grand en tokens, entièrement diffusable avec attribution, et chaque document garde sa langue et sa source,
ce qui permet une loss séparée par langue. Le tokenizer commun divise par deux le nombre de tokens du français.

**Limites.** Le web français n'a pas d'équivalent du filtre éducatif anglais : les deux langues n'ont pas
la même qualité de texte. Les quasi-doublons n'ont pas encore été mesurés sur le nouveau corpus.

**Reproduire.** Job Dagster `corpus_brut`, assets `corpus_melange` et `tokenizer_fr_en` ; modules `corpus/pretrain_sources.py` et `corpus/mix.py`.
