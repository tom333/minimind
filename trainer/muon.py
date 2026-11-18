"""
Muon + AdamW, implémentés from scratch (vérifiés contre torch.optim.Muon, cf. test en bas de fichier).

IDÉE DE MUON
------------
Pour une matrice de poids W (ex. q_proj 768×768), le gradient G a une décomposition SVD G = U S Vᵀ.
En pratique S est très déséquilibré : quelques "directions" dominent et absorbent presque toute la
mise à jour, les autres n'apprennent quasiment pas. Muon remplace G par sa version ORTHOGONALISÉE
U Vᵀ (toutes les valeurs singulières mises à ~1) : chaque direction bouge avec la même amplitude.

Calculer une SVD à chaque step serait trop cher → on approche U Vᵀ par l'itération de Newton-Schulz
(5 multiplications de matrices en bf16, très rapide sur GPU).

QUOI OPTIMISER AVEC QUOI
------------------------
- Muon : matrices 2D des couches cachées (q/k/v/o_proj, gate/up/down_proj).
- AdamW : tout le reste — embedding (partagé avec lm_head ici), gains des RMSNorm. L'orthogonalisation
  n'a pas de sens pour une table d'embedding (chaque ligne = un token indépendant) ni pour un vecteur.

LR : Muon utilise des LR ~30× plus grands qu'AdamW (0.02 typique), car sa mise à jour est normalisée.
Chaque groupe porte un 'lr_ratio' : la boucle d'entraînement fixe lr = lr_planifié × lr_ratio.
"""

import math
import torch
from torch.optim import Optimizer

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # coefficients de Keller Jordan (pente max en 0)


def zeropower_newtonschulz(G, steps=5, eps=1e-7):
    """Approche U Vᵀ (G = U S Vᵀ) par itération de Newton-Schulz quintique.
    Résultat ≈ U S' Vᵀ avec S' ∈ [0.5, 1.5] au lieu de 1 exactement : suffisant en pratique."""
    a, b, c = NS_COEFFS
    X = G.bfloat16()  # G : [lignes, colonnes] ou un lot de matrices [bloc, lignes, colonnes]
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT  # travailler sur la forme "large" (moins cher)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)  # valeurs singulières ≤ 1 pour que l'itération converge
    for _ in range(steps):
        A = X @ X.mT
        X = (
            a * X + (b * A + c * A @ A) @ X
        )  # polynôme impair en X : pousse chaque σ vers ~1
    return X.mT if transposed else X


class MuonAdamW(Optimizer):
    """Un seul optimiseur, deux règles : groupes use_muon=True → Muon, sinon → AdamW."""

    def __init__(self, param_groups):
        for g in param_groups:
            g.setdefault("lr_ratio", 1.0)
            g.setdefault("weight_decay", 0.0)
            if g["use_muon"]:
                g.setdefault("momentum", 0.95)
            else:
                g.setdefault("betas", (0.9, 0.999))
                g.setdefault("eps", 1e-8)
        super().__init__(param_groups, defaults={"lr": 1e-3})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                (self._muon_step if g["use_muon"] else self._adamw_step)(p, g)

    def _muon_step(self, p, g):
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        buf = state["momentum_buffer"]
        buf.mul_(g["momentum"]).add_(p.grad)  # B = μB + g
        update = p.grad.add(buf, alpha=g["momentum"])  # Nesterov : g + μB
        O = zeropower_newtonschulz(update)
        lr = g["lr"] * math.sqrt(
            max(1, p.size(-2) / p.size(-1))
        )  # matrices rectangulaires : même échelle
        p.mul_(1 - g["lr"] * g["weight_decay"])  # weight decay découplé
        p.add_(O.to(p.dtype), alpha=-lr)

    def _adamw_step(self, p, g):
        state = self.state[p]
        if not state:
            state["step"] = 0
            state["m"] = torch.zeros_like(p)  # moyenne mobile des gradients (inertie)
            state["v"] = torch.zeros_like(
                p
            )  # moyenne mobile des gradients² (échelle par poids)
        b1, b2 = g["betas"]
        state["step"] += 1
        state["m"].lerp_(p.grad, 1 - b1)
        state["v"].lerp_(p.grad.square(), 1 - b2)
        m_hat = state["m"] / (
            1 - b1 ** state["step"]
        )  # correction du biais d'initialisation à 0
        v_hat = state["v"] / (1 - b2 ** state["step"])
        p.mul_(1 - g["lr"] * g["weight_decay"])
        p.addcdiv_(m_hat, v_hat.sqrt().add_(g["eps"]), value=-g["lr"])


def build_muon_adamw(model, adam_lr, muon_lr, weight_decay):
    """Répartit les paramètres : matrices des blocs → Muon ; embedding et normes → AdamW."""
    muon, adam_decay, adam_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and "embed_tokens" not in name and "lm_head" not in name:
            muon.append(p)
        elif p.dim() >= 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
    return MuonAdamW(
        [
            {
                "params": muon,
                "use_muon": True,
                "lr_ratio": muon_lr / adam_lr,
                "weight_decay": weight_decay,
            },
            {"params": adam_decay, "use_muon": False, "weight_decay": weight_decay},
            {"params": adam_no_decay, "use_muon": False, "weight_decay": 0.0},
        ]
    )


if __name__ == "__main__":
    # Vérification : notre Muon et notre AdamW doivent suivre les implémentations officielles de PyTorch.
    torch.manual_seed(0)
    for shape in [(768, 768), (2432, 768), (384, 768)]:
        w0 = torch.randn(shape)
        ours, ref = w0.clone().requires_grad_(), w0.clone().requires_grad_()
        opt_ours = MuonAdamW(
            [{"params": [ours], "use_muon": True, "weight_decay": 0.1, "lr": 0.02}]
        )
        opt_ref = torch.optim.Muon([ref], lr=0.02, weight_decay=0.1)
        for _ in range(5):
            grad = torch.randn(shape)
            ours.grad, ref.grad = grad.clone(), grad.clone()
            opt_ours.step()
            opt_ref.step()
        print(
            f"Muon  {shape}: écart max avec torch.optim.Muon = {(ours - ref).abs().max():.2e}"
        )
    w0 = torch.randn(100, 50)
    ours, ref = w0.clone().requires_grad_(), w0.clone().requires_grad_()
    opt_ours = MuonAdamW(
        [{"params": [ours], "use_muon": False, "weight_decay": 0.1, "lr": 1e-3}]
    )
    opt_ref = torch.optim.AdamW([ref], lr=1e-3, weight_decay=0.1)
    for _ in range(5):
        grad = torch.randn(100, 50)
        ours.grad, ref.grad = grad.clone(), grad.clone()
        opt_ours.step()
        opt_ref.step()
    print(
        f"AdamW (100, 50): écart max avec torch.optim.AdamW = {(ours - ref).abs().max():.2e}"
    )
