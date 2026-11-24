"""
Remplir l'embedding par le calcul, sans entraînement.

Une passe de comptage sur le corpus donne, pour chaque paire de tokens, le nombre de fois où ils sont voisins
(à moins de `fenetre` tokens). On en tire la matrice d'information mutuelle (PPMI), dont la décomposition en
valeurs singulières fournit un vecteur par token : deux tokens qui ont les mêmes voisins reçoivent des vecteurs
proches (Levy et Goldberg 2014 : c'est ce que word2vec apprend par gradient).
Sortie : un fichier de poids ne contenant que l'embedding (partagé avec la sortie), à donner à l'entraînement
par --from_weight ; les blocs gardent leur initialisation ordinaire.

Usage : python embedding_comptage.py --sortie ../out/heritage/emb-comptage_768.pth
"""

import argparse
import os

import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--donnees", default="../dataset/fr_en/pretrain_fr65.jsonl")
    parser.add_argument("--vocab", type=int, default=16000)
    parser.add_argument("--largeur", type=int, default=768)
    parser.add_argument("--tokens", type=int, default=200_000_000, help="tokens du corpus lus pour compter")
    parser.add_argument("--fenetre", type=int, default=4)
    parser.add_argument("--ecart_type", type=float, default=0.02, help="échelle des vecteurs, celle de l'initialisation ordinaire")
    parser.add_argument("--sortie", required=True)
    args = parser.parse_args()

    tokens = np.load(f"{args.donnees}.train_rand2000.npy", mmap_mode="r")
    V, total = args.vocab, min(args.tokens, len(tokens))
    counts = np.zeros(V * V, dtype=np.float64)
    for start in range(0, total, 20_000_000):  # le corpus est rangé par source : on prend des morceaux répartis
        a = (len(tokens) - 20_000_000) * start // max(1, total)
        chunk = tokens[a : a + 20_000_000].astype(np.int64)
        for d in range(1, args.fenetre + 1):
            counts += np.bincount(chunk[:-d] * V + chunk[d:], minlength=V * V) / d  # voisins proches pesés plus lourd
    C = counts.reshape(V, V)
    C = C + C.T
    row = C.sum(1, keepdims=True)
    pmi = np.log(np.maximum(C * C.sum() / np.maximum(row * row.T, 1e-12), 1e-12))
    ppmi = torch.from_numpy(np.maximum(pmi, 0.0)).float()
    seen = int((row[:, 0] > 0).sum())
    U, S, _ = torch.svd_lowrank(ppmi, q=args.largeur + 64, niter=4)
    E = U[:, : args.largeur] * S[: args.largeur].sqrt()
    E = E * (args.ecart_type / E.std())
    print(f"{total / 1e6:.0f}M tokens comptés, {seen} tokens vus sur {V} ; valeurs singulières : {S[0]:.0f} … {S[args.largeur - 1]:.1f}")
    os.makedirs(os.path.dirname(args.sortie), exist_ok=True)
    torch.save({"model.embed_tokens.weight": E.half(), "lm_head.weight": E.half()}, args.sortie)
    print(args.sortie)
