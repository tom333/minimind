"""Propriétés du modèle qui doivent rester vraies quand on le modifie."""

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.ternary import TernaryLinear

SMALL = dict(vocab_size=6400, hidden_size=64, num_hidden_layers=4)


def test_un_pas_donne_une_loss_finie():
    model = MiniMindForCausalLM(MiniMindConfig(**SMALL))
    x = torch.randint(0, 6400, (2, 32))
    out = model(x, labels=x)
    out.loss.backward()
    assert torch.isfinite(out.loss) and abs(out.loss.item() - 8.76) < 0.5  # hasard : ln(6400) = 8,76


def test_mod_les_deux_chemins_donnent_le_meme_resultat():
    """Mixture-of-Depths : le chemin rapide (entraînement) et le chemin causal (évaluation) coïncident
    quand ils retiennent les mêmes tokens."""
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig(mod_capacity=0.5, **SMALL)).float()
    block, length = model.model.layers[1], 40
    h = torch.randn(1, length, 64)
    pos = (model.model.freqs_cos[:length], model.model.freqs_sin[:length])
    with torch.no_grad():
        block.router.weight.normal_()
        kept = int((block.router(h).squeeze(-1) > 0).sum())
        block.mod_capacity = kept / length + 1e-6
        block.train()
        fast, _ = block(h, pos)
        block.eval()
        causal, _ = block(h, pos)
    assert 0 < kept < length and (fast - causal).abs().max() < 1e-4


def test_mod_ordre_sans_tri_identique_au_tri():
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for length in (340, 768, 2048):
            score, k = torch.randn(4, length).to(dtype), length // 4
            top = score.topk(k, dim=1).indices
            target = torch.zeros_like(score).scatter(1, top, 1.0)
            rank = torch.arange(length, 0, -1, dtype=torch.float32)
            order = (target.float() * rank).topk(k, dim=1).indices
            assert (order == top.sort(dim=1).values).all()


def test_ternaire_lambda_zero_reproduit_le_modele_precis():
    torch.manual_seed(0)
    precise = MiniMindForCausalLM(MiniMindConfig(**SMALL))
    ternary = MiniMindForCausalLM(MiniMindConfig(ternary="row", **SMALL))
    ternary.load_state_dict(precise.state_dict())
    layers = [m for m in ternary.modules() if isinstance(m, TernaryLinear)]
    x = torch.randint(0, 6400, (1, 16))
    for m in layers:
        m.lam.fill_(0.0)
    assert (ternary(x).logits - precise(x).logits).abs().max() < 1e-5
    for m in layers:
        m.lam.fill_(1.0)
    assert (ternary(x).logits - precise(x).logits).abs().max() > 1e-3
    assert all(set(m.trits()[0].unique().tolist()) <= {-1.0, 0.0, 1.0} for m in layers)


def test_diffusion_masquee_loss_finie_et_reproductible():
    from trainer.diffusion import mdm_loss, set_causal

    model = MiniMindForCausalLM(MiniMindConfig(**SMALL))
    set_causal(model, False)
    x = torch.randint(30, 6400, (4, 32))
    a = mdm_loss(model, x, generator=torch.Generator().manual_seed(0))
    b = mdm_loss(model, x, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(a) and a.item() == b.item()


def test_ngrammes_ne_regardent_pas_le_futur():
    """Tables de n-grammes : changer un token ne modifie aucune sortie des positions précédentes."""
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig(ngram_orders=(2, 3, 4), ngram_rows=1024, ngram_dim=16, **SMALL)).eval()
    x = torch.randint(0, 6400, (2, 32))
    y = x.clone()
    y[:, 20] = (y[:, 20] + 1) % 6400
    with torch.no_grad():
        a, b = model(x).logits, model(y).logits
    assert (a[:, :20] - b[:, :20]).abs().max() < 1e-5 and (a[:, 20:] - b[:, 20:]).abs().max() > 1e-4


def test_heritage_par_regression_retrouve_le_parent():
    """Moindres carrés bloc par bloc : avec une tranche par bloc, on doit retrouver la matrice effacée du parent."""
    import copy

    from trainer.heritage_regression import fill_by_least_squares

    torch.manual_seed(0)
    parent = MiniMindForCausalLM(MiniMindConfig(**SMALL)).float().eval()
    layers = list(parent.model.layers)
    child = [copy.deepcopy(layer) for layer in layers]
    for layer in child:  # on efface ce que la régression doit retrouver
        layer.mlp.down_proj.weight.data.zero_()
    fill_by_least_squares(parent, child, list(range(len(layers) + 1)), torch.randint(0, 6400, (64, 48)), ridge=1e-9)
    for mine, ref in zip(child, layers):
        for a, b in ((mine.self_attn.o_proj, ref.self_attn.o_proj), (mine.mlp.down_proj, ref.mlp.down_proj)):
            assert (a.weight - b.weight).norm() / b.weight.norm() < 0.05


def test_grilles_3d_a_la_place_du_ffn():
    """Grilles 3D : le modèle tourne, et les grilles comme les projections qui choisissent où lire reçoivent un gradient."""
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig(grid_layers=(0, 1, 2, 3), grid_count=4, grid_features=4, grid_sizes=(4, 8), **SMALL))
    x = torch.randint(0, 6400, (2, 32))
    model(x, labels=x).loss.backward()
    grid_ffn = model.model.layers[0].mlp
    assert grid_ffn.mem_grids[0].grad.abs().sum() > 0 and grid_ffn.coords.weight.grad.abs().sum() > 0


def test_reduction_de_largeur_exacte_sans_perte():
    """Rotation du flux résiduel + poids des normalisations fondus dans les matrices : sans rien couper (même
    largeur, tous les neurones, tous les blocs), l'enfant calcule exactement la même chose que le parent."""
    from trainer.heritage_largeur import carve_width

    torch.manual_seed(0)
    config = MiniMindConfig(**SMALL)
    parent = MiniMindForCausalLM(config).float().eval()
    for layer in parent.model.layers:  # des normalisations non triviales, à fondre dans les matrices
        layer.input_layernorm.weight.data.uniform_(0.5, 1.5)
        layer.post_attention_layernorm.weight.data.uniform_(0.5, 1.5)
    x = torch.randint(0, 6400, (4, 40))
    keep = list(range(config.num_hidden_layers))
    state, _ = carve_width(parent, keep, config.hidden_size, config.intermediate_size, x, "cpu")
    child = MiniMindForCausalLM(config).float().eval()
    child.load_state_dict(state)
    with torch.no_grad():
        assert (child(x).logits - parent(x).logits).abs().max() < 1e-3
