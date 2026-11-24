"""
Héritage : tailler un petit modèle dans un grand modèle déjà entraîné, au lieu de partir de poids aléatoires.

Le parent doit avoir la même architecture que MiniMind (famille Qwen3 : mêmes blocs, mêmes noms de poids).
Deux étapes :
  1. vocabulaire : le parent connaît ~152 000 tokens (toutes langues, code) ; on ne garde que ceux qui servent
     dans notre corpus. Les fusions BPE gardées le sont dans leur ordre d'origine, avec tous leurs ancêtres : un
     mot dont les tokens sont tous gardés se découpe exactement comme chez le parent ; les autres se redécoupent
     en morceaux plus petits.
  2. tailler : lignes de l'embedding des tokens gardés + sous-ensemble de blocs. Les blocs sont retirés un à un,
     en enlevant à chaque tour celui dont l'absence dégrade le moins la loss (mesurée sur un extrait du corpus).
Sorties : un dossier tokenizer, et des poids au format du projet (<nom>_<largeur>.pth) avec leur fiche
(<nom>.json : parent, forme du modèle, tokenizer, blocs gardés), lisible par eval_cross.py (nom=poids.pth@fiche.json).

Usage :
  python heritage.py vocabulaire Qwen/Qwen3-0.6B-Base --taille 24000 --sortie ../dataset/fr_en/tokenizer_q3_24k
  python heritage.py tailler Qwen/Qwen3-0.6B-Base --tokenizer ../dataset/fr_en/tokenizer_q3_24k --blocs 8 --nom q3-06-b8
"""

import argparse
import glob
import itertools
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F

DATA_DIR = os.environ.get("MINIMIND_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "dataset"))
SEQ = 340


def corpus_sample(path, pas, decalage=0):
    """Un document sur `pas` : le fichier est rangé par source, un pas régulier respecte donc ses proportions."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for i, line in enumerate(f) if i % pas == decalage]


def encode(tok, texts):
    """Liste de listes d'ids, par lots (le tokenizer rapide travaille en parallèle sur un lot)."""
    out = []
    for i in range(0, len(texts), 2000):
        out += tok(texts[i : i + 2000], add_special_tokens=False).input_ids
    return out


def cmd_vocabulaire(args):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.parent)
    docs = encode(tok, corpus_sample(args.corpus, args.pas))
    counts = np.bincount(np.fromiter(itertools.chain.from_iterable(docs), dtype=np.int64), minlength=len(tok))

    tj = json.load(open(hf_hub_download(args.parent, "tokenizer.json"), encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    merges = [m.split(" ") if isinstance(m, str) else m for m in tj["model"]["merges"]]
    parents = {}  # token fusionné → toutes les paires qui le produisent
    for a, b in merges:
        parents.setdefault(vocab[a + b], []).append((vocab[a], vocab[b]))
    base = [i for i in vocab.values() if i not in parents]  # les 256 octets
    order = np.argsort(-counts[: len(vocab)], kind="stable")

    def closure(m):
        """Les m tokens les plus fréquents, les octets, et tous les ancêtres nécessaires à leurs fusions."""
        kept, stack = set(base), [int(i) for i in order[:m]]
        while stack:
            i = stack.pop()
            if i not in kept:
                kept.add(i)
                stack += [p for pair in parents.get(i, ()) for p in pair]
        return kept

    specials = tj["added_tokens"]
    lo, hi = 0, len(vocab)
    while lo < hi:  # le plus grand m dont la fermeture tient dans la taille demandée
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if len(closure(mid)) + len(specials) <= args.taille else (lo, mid - 1)
    kept = sorted(closure(lo))
    names = {i: t for t, i in vocab.items()}
    new_vocab = {names[i]: n for n, i in enumerate(kept)}
    new_merges = [[a, b] for a, b in merges if a in new_vocab and b in new_vocab and a + b in new_vocab]
    old_ids = kept + [s["id"] for s in specials]
    for n, s in enumerate(specials):
        s["id"] = len(kept) + n
    tj["model"]["vocab"] = new_vocab
    tj["model"]["merges"] = [" ".join(m) for m in new_merges] if isinstance(tj["model"]["merges"][0], str) else new_merges

    os.makedirs(args.sortie, exist_ok=True)
    json.dump(tj, open(os.path.join(args.sortie, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    sep = tok.eos_token  # le parent n'a pas de token de début : le séparateur de documents en tient lieu
    json.dump(
        dict(tokenizer_class="PreTrainedTokenizerFast", bos_token=sep, eos_token=sep, pad_token=sep, model_max_length=131072, clean_up_tokenization_spaces=False),
        open(os.path.join(args.sortie, "tokenizer_config.json"), "w"),
        indent=1,
    )
    json.dump(dict(parent=args.parent, ids=old_ids), open(os.path.join(args.sortie, "heritage.json"), "w"))

    # Contrôle sur d'autres documents : le découpage est-il celui du parent là où tous ses tokens sont gardés ?
    small = AutoTokenizer.from_pretrained(args.sortie)
    texts = corpus_sample(args.corpus, args.pas * 20, decalage=1)
    new_of = {old: n for n, old in enumerate(old_ids)}
    before, after, same, checked = 0, 0, 0, 0
    for old, new in zip(encode(tok, texts), encode(small, texts)):
        before, after = before + len(old), after + len(new)
        if all(i in new_of for i in old):
            checked += 1
            same += [new_of[i] for i in old] == new
    print(f"tokens du parent : {len(tok)} → gardés : {len(small)} (dont {len(specials)} spéciaux, {len(new_merges)} fusions)")
    print(f"occurrences couvertes par les tokens gardés : {counts[kept].sum() / counts.sum():.4%} (sur {counts.sum() / 1e6:.1f}M tokens)")
    print(f"documents de contrôle : {len(texts)} ; entièrement couverts : {checked}, dont découpage identique : {same}")
    print(f"longueur des textes en tokens, vocabulaire réduit / parent : {after / before:.4f} (1 = identique)")


def calibration(tok, path, n_seq, pas, decalage):
    bos, eos = tok.bos_token_id, tok.eos_token_id
    ids = [t for doc in encode(tok, corpus_sample(path, pas, decalage)) for t in [bos] + doc + [eos]]
    return torch.tensor(ids[: n_seq * SEQ]).view(-1, SEQ)


@torch.no_grad()
def loss_on(model, blocks, device):
    nll = 0.0
    for b in blocks.split(32):
        b = b.to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(b).logits[:, :-1].float()
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), b[:, 1:].reshape(-1), reduction="sum").item()
    return nll / blocks[:, 1:].numel()


def load_parent(parent, tokenizer_dir, device):
    """Le parent entier dans notre classe de modèle, avec le vocabulaire réduit. Retourne (modèle, tokenizer,
    forme, ids d'origine des tokens gardés)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    meta = json.load(open(os.path.join(tokenizer_dir, "heritage.json")))
    assert meta["parent"] == parent, "ce tokenizer a été réduit pour un autre parent"
    folder = snapshot_download(parent, allow_patterns=["*.safetensors", "config.json"])
    cfg = json.load(open(os.path.join(folder, "config.json")))
    state = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        state.update(load_file(f))
    ids = torch.tensor(meta["ids"])
    embedding = state["model.embed_tokens.weight"][ids]
    state["lm_head.weight"] = state["lm_head.weight"][ids] if "lm_head.weight" in state else embedding
    state["model.embed_tokens.weight"] = embedding
    shape = {k: cfg[k] for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "rope_theta", "rms_norm_eps", "tie_word_embeddings")}
    model = MiniMindForCausalLM(MiniMindConfig(vocab_size=len(tok), **shape))
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    assert not missing and not unexpected, f"poids non appariés : {missing} {unexpected}"
    return model.to(device).eval(), tok, shape, ids


def save_child(model, layers, shape, sortie, nom, fiche):
    """Écrit l'enfant (embedding, norme finale et tête du modèle donné + les blocs listés) au format du projet."""
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    shape = {**shape, "num_hidden_layers": len(layers)}
    child = MiniMindForCausalLM(MiniMindConfig(vocab_size=model.config.vocab_size, **shape))
    state = {k: v for k, v in model.state_dict().items() if ".layers." not in k}
    for n, layer in enumerate(layers):
        state.update({f"model.layers.{n}.{k}": v for k, v in layer.state_dict().items()})
    child.load_state_dict(state)
    os.makedirs(sortie, exist_ok=True)
    path = os.path.join(sortie, f"{nom}_{shape['hidden_size']}.pth")
    torch.save({k: v.half().cpu() for k, v in child.state_dict().items()}, path)
    json.dump({**fiche, "config": shape}, open(os.path.join(sortie, f"{nom}.json"), "w"), indent=1)
    print(f"{path} : {sum(p.numel() for p in child.parameters()) / 1e6:.1f}M paramètres")


def cmd_tailler(args):
    from transformers import AutoTokenizer

    model, tok, shape, ids = load_parent(args.parent, args.tokenizer, args.device)
    meta = {"ids": ids.tolist()}

    if args.verifier:  # notre code calcule-t-il la même chose que le code d'origine du parent ?
        from transformers import AutoModelForCausalLM

        ref_tok = AutoTokenizer.from_pretrained(args.parent)
        ref = AutoModelForCausalLM.from_pretrained(args.parent, dtype=torch.float32).to(args.device).eval()
        new_of = {old: n for n, old in enumerate(meta["ids"])}
        docs = encode(ref_tok, corpus_sample(os.path.join(DATA_DIR, args.corpus), args.pas, decalage=4))
        old_ids = next(d[:200] for d in docs if len(d) >= 200 and all(i in new_of for i in d[:200]))  # extrait entièrement couvert
        a = torch.tensor([old_ids], device=args.device)
        b = torch.tensor([[new_of[i] for i in old_ids]], device=args.device)
        with torch.no_grad():
            gap = (ref(a).logits[0][:, ids.to(args.device)] - model(b).logits[0]).abs().max().item()
        print(f"écart maximal des logits avec le code d'origine : {gap:.5f} (attendu : < 0,01)")
        del ref
        torch.cuda.empty_cache()

    corpus = os.path.join(DATA_DIR, args.corpus)
    calib = calibration(tok, corpus, args.sequences, args.pas, decalage=2)
    control = calibration(tok, corpus, args.sequences, args.pas, decalage=3)  # jamais vu par le choix des blocs
    layers = list(model.model.layers)

    def with_layers(keep, blocks):
        model.model.layers = torch.nn.ModuleList([layers[i] for i in keep])
        return loss_on(model, blocks, args.device)

    keep = list(range(len(layers)))
    print(f"{len(keep)} blocs (parent, vocabulaire réduit) : loss {with_layers(keep, calib):.3f} ; contrôle {with_layers(keep, control):.3f}")
    if args.liste:
        keep = [int(i) for i in args.liste.split(",")]
    while len(keep) > args.blocs:
        loss, out = min((with_layers([i for i in keep if i != j], calib), j) for j in keep)
        keep.remove(out)
        print(f"{len(keep)} blocs : retrait du bloc {out}, loss {loss:.3f}", flush=True)
    print(f"blocs gardés : {keep} ; loss {with_layers(keep, calib):.3f} ; contrôle {with_layers(keep, control):.3f} (hasard : {np.log(len(tok)):.2f})")

    fiche = dict(parent=args.parent, blocs_gardes=keep, tokenizer=os.path.relpath(args.tokenizer, DATA_DIR))
    save_child(model, [layers[i] for i in keep], shape, args.sortie, args.nom, fiche)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="commande", required=True)
    p = sub.add_parser("vocabulaire", help="réduit le vocabulaire du parent aux tokens utiles dans notre corpus")
    p.add_argument("parent", help="dépôt Hugging Face du parent, ex. Qwen/Qwen3-0.6B-Base")
    p.add_argument("--corpus", default=os.path.join(DATA_DIR, "fr_en", "pretrain_fr65hq.jsonl"))
    p.add_argument("--pas", type=int, default=50, help="un document sur `pas` sert à compter les tokens")
    p.add_argument("--taille", type=int, default=24000, help="nombre de tokens gardés, spéciaux compris")
    p.add_argument("--sortie", required=True)
    p.set_defaults(run=cmd_vocabulaire)
    p = sub.add_parser("tailler", help="garde les lignes d'embedding du vocabulaire réduit et un sous-ensemble de blocs")
    p.add_argument("parent")
    p.add_argument("--tokenizer", required=True, help="dossier produit par la commande vocabulaire")
    p.add_argument("--blocs", type=int, default=8, help="nombre de blocs gardés")
    p.add_argument("--liste", default="", help="blocs de départ imposés, ex. '0,1,2,25,26,27' (défaut : tous)")
    p.add_argument("--corpus", default="fr_en/pretrain_fr65hq.jsonl", help="relatif au dossier des données")
    p.add_argument("--pas", type=int, default=5000, help="un document sur `pas` sert à choisir les blocs")
    p.add_argument("--sequences", type=int, default=96, help="séquences de 340 tokens pour mesurer la loss")
    p.add_argument("--sortie", default="../out/heritage")
    p.add_argument("--nom", required=True)
    p.add_argument("--verifier", action="store_true", help="compare nos logits à ceux du code d'origine du parent")
    p.add_argument("--device", default="cuda")
    p.set_defaults(run=cmd_tailler)
    args = parser.parse_args()
    args.run(args)
