# Études

Journal des expériences menées sur ce dépôt, du 16 au 26 novembre 2026. Chaque fiche répond à une
question, avec son protocole, ses mesures, sa conclusion et ses limites. Les résultats négatifs y figurent
au même titre que les autres.

## Règles communes

- **Matériel** : une RTX 3060 (12 Go). Un essai standard dure 20 minutes (50M tokens).
- **Mesure** : loss de validation sur 2 000 documents tirés au hasard, jamais vus à l'entraînement. Plus bas = mieux.
- **Bruit** : 0,012 de val_loss entre deux graines avec la recette finale. Un écart plus petit n'est pas interprété.
- **Repères** : hasard 8,76 ; fréquences des tokens seules 7,40 ; réglages d'origine à 50M tokens 2,95.
- **Budget** : 20 minutes suffisent pour juger une optimisation. Un ajout de capacité se juge sur un run long (étude 06).
- **Suivi** : chaque run est enregistré dans MLflow, chaque campagne dans une étude Optuna.

## Synthèse

| # | Question | Résultat | Verdict |
|---|---|---|---|
| [01](01-debit-entrainement.md) | Où part le temps de calcul ? | débit utile ×3 sans toucher au modèle ; 53 % du calcul portait sur du remplissage | gain |
| [02](02-validation-et-bruit.md) | Peut-on se fier à la mesure ? | la validation initiale ne contenait que de l'anglais ; bruit mesuré à 0,012 | correction |
| [03](03-learning-rate-et-reglages.md) | Réglages et retouches publiées | learning rate d'origine déjà bon ; WSD, plafonnement des logits et initialisation à zéro n'apportent rien | négatif |
| [04](04-taille-de-batch.md) | Taille de batch | −0,42 de val_loss en divisant le batch par quatre | gain majeur |
| [05](05-optimiseur-muon.md) | Muon contre AdamW | −0,14 pour 15 % de temps en plus | gain |
| [06](06-loi-d-echelle.md) | Que rapporte un token de plus ? | run long à 2,006, dans la plage prédite (1,89 à 2,18) | prédiction vérifiée |
| [07](07-architecture.md) | Profondeur, largeur, têtes, experts | aucune variante ne bat l'architecture d'origine | négatif |
| [08](08-matrices-structurees.md) | Matrices structurées dans le FFN | moins bon de 0,058, mémoire ×2,5, temps ×2,4 | négatif |
| [09](09-couche-memoire.md) | Couche mémoire à clés produit | −0,010 pour +80 % de paramètres | négatif |
| [10](10-compression-des-poids.md) | Redondance des poids | 2 fois plus petit sans perte mesurable ; ni voisinage, ni neurones morts | constat |
| [11](11-poids-ternaires.md) | Poids à trois valeurs | voir la fiche | gain sous condition |
| [12](12-ajustement-conversation.md) | Du modèle de langue à l'assistant | val_loss 2,38 → 1,26 ; réglage d'origine dix fois trop bas | gain |
| [13](13-evaluation-et-raisonnement.md) | Au-delà de la moyenne | tokens rares deux fois plus durs ; 10 % de réussite sur deux opérations enchaînées | constat |
| [14](14-corpus.md) | Les données | corpus bilingue de 1,02 milliard de tokens, traçable et diffusable | construction |
| [15](15-plateforme.md) | La plateforme | Dagster, Optuna et MLflow, avec partage automatique du GPU | construction |
| [16](16-poids-ternaires-suite.md) | Ternaire : recette et agrandissement | seul le taux d'apprentissage compte (×2) ; 104M ternaire comble la moitié de l'écart à 3× moins de mémoire | gain partiel |
| [17](17-acceleration.md) | Accélérer l'entraînement | aucun gain gratuit ; à temps égal, rien ne bat le modèle d'origine | négatif |
| [18](18-mixture-of-depths.md) | Router les tokens | +33 % de débit pour un peu de loss ; défaut de compilation de 15 min trouvé et contourné | gain sous condition |
| [19](19-modele-150m.md) | Vers 150M | −0,087 de loss dès 50M tokens ; 14 à 20 h par milliard sur la RTX 3060 | constat |
| [20](20-gpu-externes.md) | GPU externes | fp16 sur T4 sans perte ; Kaggle, Colab, distribué, SkyPilot | construction |
| [21](21-diffusion-et-hybride.md) | Diffusion et hybride | mécanisme fonctionnel ; ni plus rapide ni meilleur à cette taille | négatif à cette échelle |
| [22](22-industrialisation.md) | Industrialisation | traçabilité, tests, registre, bancs standards | construction |
| [23](23-distillation-repetition-generation.md) | Distillation, répétition, génération | distillation −0,095 à tokens égaux, nulle à temps égal ; 3 relectures +0,093 ; pénalité de répétition 1,2 adoptée | gain sous condition |
| [24](24-tables-de-ngrammes.md) | Tables de n-grammes à l'entrée | −0,018 pour un coût nul ; l'avance fond pendant l'entraînement | gain faible |
| [25](25-heritage.md) | Hériter des poids d'un modèle existant | un 150M taillé dans Qwen3-0.6B dépasse notre 71M de 8 h en 2 h | gain majeur |
| [26](26-comptage.md) | Que remplace le comptage ? | le comptage vaut 9 minutes d'entraînement, mais ajoute −0,045 bit/octet au modèle de 8 h | gain |
| [27](27-negatifs-revisites.md) | Résultats négatifs, testés autrement | distillation hors ligne et réduction de largeur se rachètent ; embedding par comptage −0,033 à 200M tokens ; boucle, hybride, comptage pendant l'entraînement : négatifs | gain partiel |
| [28](28-ce-que-l-heritage-transmet.md) | Ce que l'héritage transmet | la langue oui, le raisonnement non : la coupe, les données et le contexte l'ont jeté ; le parent résout 53 % du calcul d'école | constat |
| [29](29-300m-profond-raisonnement.md) | Un 300M profond qui calcule | 24,8 % au calcul d'école (SmolLM2-360M : 4,8 %), mais 2 % en français et compréhension au hasard | gain partiel |

## Le chemin parcouru, à budget égal (50M tokens)

| Étape | val_loss |
|---|---|
| Réglages d'origine | 2,95 |
| + warmup et weight decay | 2,91 |
| + batch de 64 séquences au lieu de 256 | 2,49 |
| + optimiseur Muon | 2,35 |

Ce gain de 0,60 dépasse celui d'un doublement des données (0,49 entre 50M et 100M tokens). Dans le même
temps, sur six techniques tirées de publications récentes, une seule a tenu à cette échelle : Muon. Les cinq
autres (noyaux Liger, calendrier WSD, plafonnement des logits avec initialisation à zéro, matrices
structurées, couche mémoire) n'apportent rien de mesurable ou dégradent le modèle.

## Crédit

Le modèle et les scripts de base viennent de [MiniMind](https://github.com/jingyaogong/minimind) (Apache 2.0).
Les études, l'outillage d'expérimentation, la chaîne de données bilingue et les variantes d'architecture
testées ici sont propres à ce dépôt.
