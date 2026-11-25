"""
Distillation hors ligne : les probabilités de l'enseignant sont calculées UNE fois et rangées sur disque.

Avec la distillation en ligne, l'enseignant refait un calcul complet à chaque pas, et ses probabilités prennent
tant de mémoire qu'il faut réduire le lot : l'essai dure 42 min au lieu de 27, ce qui efface le gain à temps égal.
Ici, l'enseignant passe une seule fois sur les séquences du sous-ensemble d'entraînement (--train_subset_tokens) ;
pour chaque position on garde ses k meilleurs tokens et leurs probabilités (renormalisées sur ces k, exactement
comme la distillation en ligne à température 1). L'entraînement les lit ensuite comme les tokens eux-mêmes.
Fichiers : ids.npy (uint16) et probs.npy (float16), [séquences, longueur - 1, k], puis meta.json, écrit en dernier.
"""

import json
import os
import time

import numpy as np
import torch
from numpy.lib.format import open_memmap


def build_or_load(args, train_ds, vocab_size, log):
    """Retourne (ids, probs) en lecture seule ; les calcule d'abord s'ils n'existent pas pour ce sous-ensemble."""
    meta = dict(teacher=os.path.abspath(args.teacher), data=os.path.abspath(args.data_path), subset_tokens=args.train_subset_tokens,
                seq=args.max_seq_len, topk=args.distill_topk, sequences=len(train_ds))
    folder = args.teacher_cache
    paths = [os.path.join(folder, f) for f in ("ids.npy", "probs.npy", "meta.json")]
    if os.path.exists(paths[2]):
        found = json.load(open(paths[2]))
        found.pop("minutes", None)
        assert found == meta, f"cache d'enseignant {folder} fait pour un autre essai : {found} ≠ {meta}"
        log(f"distillation hors ligne : probabilités de l'enseignant lues dans {folder}")
        return np.load(paths[0], mmap_mode="r"), np.load(paths[1], mmap_mode="r")

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    start = time.time()
    teacher = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_layers, vocab_size=vocab_size))
    teacher.load_state_dict({k: v.float() for k, v in torch.load(args.teacher, map_location="cpu").items()})
    teacher = teacher.to(args.device).eval()
    os.makedirs(folder, exist_ok=True)
    shape = (len(train_ds), args.max_seq_len - 1, args.distill_topk)
    ids, probs = open_memmap(paths[0], "w+", np.uint16, shape), open_memmap(paths[1], "w+", np.float16, shape)
    cuda = "cuda" in args.device
    with torch.no_grad():
        for a in range(0, len(train_ds), 64):
            batch = torch.stack([train_ds[i][0] for i in range(a, min(a + 64, len(train_ds)))]).to(args.device)
            with torch.autocast("cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
                logits = teacher(batch).logits[:, :-1].float()
            top, idx = logits.topk(args.distill_topk, dim=-1)
            ids[a : a + len(batch)] = idx.to(torch.int32).cpu().numpy().astype(np.uint16)
            probs[a : a + len(batch)] = top.softmax(-1).cpu().numpy().astype(np.float16)
    ids.flush(), probs.flush()
    del teacher, ids, probs
    if cuda:
        torch.cuda.empty_cache()
    minutes = (time.time() - start) / 60
    json.dump({**meta, "minutes": minutes}, open(paths[2], "w"), indent=1)
    log(f"distillation hors ligne : probabilités de l'enseignant calculées en {minutes:.1f} min → {folder}")
    return np.load(paths[0], mmap_mode="r"), np.load(paths[1], mmap_mode="r")
