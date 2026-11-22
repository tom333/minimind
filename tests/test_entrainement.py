"""Tests de fumée de la chaîne d'entraînement, sur processeur, avec un corpus synthétique."""

import os

from conftest import train


def test_entrainement_apprend(corpus, tmp_path):
    _, losses, val = train(corpus, tmp_path / "a", "--save_resume", "0")
    assert losses[max(losses)] < losses[min(losses)] - 0.1 and val is not None


def test_reprise_apres_arret_identique_au_run_ininterrompu(corpus, tmp_path):
    """Arrêt programmé au bout d'un instant, puis reprise : mêmes losses pas à pas et même val_loss finale."""
    _, ref_losses, ref_val = train(corpus, tmp_path / "ref", "--save_resume", "0")
    out, _, _ = train(corpus, tmp_path / "b", "--save_resume", "1", "--max_hours", "0.000001")
    assert "arrêt programmé" in out and os.path.exists(tmp_path / "b" / "t_64_resume.pth")
    _, losses, val = train(corpus, tmp_path / "b", "--save_resume", "1", "--from_resume", "1")
    assert losses and all(ref_losses[s] == v for s, v in losses.items()) and val == ref_val


def test_variantes_tournent(corpus, tmp_path):
    for i, extra in enumerate((["--mod_capacity", "0.5"], ["--ternary", "row"], ["--objective", "mdm"], ["--ffn_glu", "0", "--hidden_act", "relu2"])):
        _, losses, val = train(corpus, tmp_path / f"v{i}", "--save_resume", "0", *extra, steps=8)
        assert losses and val is not None, extra


def test_serveur_de_suivi_injoignable_ne_tue_pas_l_entrainement(corpus, tmp_path):
    out, losses, val = train(corpus, tmp_path / "m", "--save_resume", "0", "--use_mlflow", "--mlflow_uri", "http://127.0.0.1:9", steps=8)
    assert "MLflow indisponible" in out and losses and val is not None
