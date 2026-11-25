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


def test_mode_distribue_deux_processus(corpus, tmp_path):
    """Deux processus sur processeur (gloo) : l'entraînement distribué va au bout et donne une loss finie."""
    import subprocess
    import sys

    from conftest import ROOT

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", HF_DATASETS_CACHE=str(tmp_path / "hf"), PYTHONPATH=ROOT)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", "train_pretrain.py",
           "--device", "cpu", "--dtype", "float32", "--data_path", corpus, "--tokenizer_path", "../model", "--packing", "1",
           "--max_seq_len", "64", "--max_steps", "6", "--eval_interval", "6", "--val_docs", "20", "--batch_size", "2",
           "--hidden_size", "64", "--num_hidden_layers", "2", "--use_compile", "0", "--optimizer", "muon",
           "--num_workers", "0", "--save_resume", "0", "--save_dir", str(tmp_path / "d"), "--save_weight", "d"]
    run = subprocess.run(cmd, cwd=os.path.join(ROOT, "trainer"), env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and "step 6: val_loss" in run.stdout, run.stdout[-1500:] + run.stderr[-1500:]


def test_arret_anticipe_garde_le_calendrier_de_l_essai_complet(corpus, tmp_path):
    """--stop_fraction : arrêt au tiers du parcours, avec évaluation finale ; le taux d'apprentissage est celui de
    l'essai complet au même pas (pas encore descendu à zéro)."""
    out, losses, val = train(corpus, tmp_path / "s", "--save_resume", "0", "--stop_fraction", "0.34", "--eval_interval", "100")
    assert max(losses) == 8 and val is not None
    _, full, _ = train(corpus, tmp_path / "f", "--save_resume", "0")
    assert all(full[s] == v for s, v in losses.items())
