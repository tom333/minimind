import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import copy
import json
import random
import numpy as np
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset, PackedPretrainDataset, SFTDataset
from trainer.diffusion import mdm_loss, set_causal
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def sft_char_lengths(samples, cache_path):
    """Longueur en caractères de chaque conversation (approximation bon marché de sa longueur en tokens)."""
    if os.path.exists(cache_path): return torch.from_numpy(np.load(cache_path))
    lengths = [sum(len(m.get('content') or '') + len(m.get('reasoning_content') or '') for m in conv)
               for batch in samples.iter(batch_size=10000) for conv in batch['conversations']]
    np.save(cache_path, np.array(lengths, dtype=np.int32))
    return torch.tensor(lengths)


def bucketed_batches(lengths, batch_size, seed):
    """Batches de conversations de longueurs voisines (tri par paquets de 100 batches) : peu de padding,
    tout en gardant un ordre aléatoire des batches."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for chunk in torch.randperm(len(lengths), generator=g).split(batch_size * 100):
        batches += list(chunk[torch.argsort(lengths[chunk])].split(batch_size))
    return [batches[i].tolist() for i in torch.randperm(len(batches), generator=g)]


def collate_sft(batch):
    """Coupe le batch à la longueur de sa plus longue conversation (arrondie à 64 : limite le nombre
    de formes différentes vues par torch.compile)."""
    input_ids, labels = torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])
    longest = int((input_ids != pad_id).sum(1).max())
    keep = min(input_ids.size(1), -(-longest // 64) * 64)
    return input_ids[:, :keep], labels[:, :keep]


def val_langs(data_path, val_docs):
    """Langue de chaque document de validation (même tirage que le split), ou None si le corpus n'a pas de
    champ `lang`. Mise en cache à côté du corpus : le fichier JSONL n'est relu qu'une fois."""
    cache = f'{data_path}.val_rand{val_docs}.langs.json'
    if os.path.exists(cache):
        with open(cache) as f: return json.load(f)
    with open(data_path, encoding='utf-8') as f: langs = [json.loads(line).get('lang') for line in f]
    idx = sorted(random.Random(0).sample(range(len(langs)), val_docs))
    out = [langs[i] for i in idx] if langs[0] is not None else None
    try:
        with open(cache, 'w') as f: json.dump(out, f)
    except OSError: pass  # dossier en lecture seule (jeu de données monté)
    return out


def track(mlflow, method, *args, **kwargs):
    """Écrit dans MLflow sans jamais interrompre l'entraînement : si le serveur de suivi est injoignable
    (redémarrage, réseau), la mesure est perdue et signalée, le calcul continue. Le journal texte garde tout."""
    if not mlflow: return
    try:
        getattr(mlflow, method)(*args, **kwargs)
    except Exception as e:
        Logger(f'MLflow indisponible ({method}) : {type(e).__name__} — mesure non enregistrée, entraînement poursuivi')


def save_weights(path):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, path)


@torch.no_grad()
def evaluate(val_loader):
    random.seed(0)  # SFT : prompt système et balises tirés au hasard par le dataset → validation reproductible
    model.eval()
    losses, bounds = [], []
    gen = torch.Generator(device=args.device).manual_seed(0)  # mêmes masques à chaque évaluation
    for input_ids, labels in val_loader:
        input_ids = input_ids.to(args.device)
        with autocast_ctx:
            if args.objective != 'mdm':
                set_causal(model, True)
                losses.append(model(input_ids, labels=labels.to(args.device)).loss.item())
            if args.objective != 'ar':
                set_causal(model, False)
                bounds.append(mdm_loss(model, input_ids, generator=gen).item())
    model.train()
    if bounds: Logger(f'borne diffusion (nats/token) : {sum(bounds) / len(bounds):.4f}')
    return sum(losses or bounds) / len(losses or bounds)  # val_loss = loss mot à mot si elle existe


eval_seconds = [0.0]  # temps passé en évaluation depuis le début de l'epoch
run_start, stopped_early = time.time(), [False]  # pour --max_hours


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    eval_seconds[0] = 0.0
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        global_step = epoch * iters + step
        last_step = step == iters or global_step >= stop_step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        warmup_steps = int(args.warmup_ratio * total_steps)
        decay_start = int((1 - args.decay_ratio) * total_steps)
        if global_step < warmup_steps:
            lr = args.learning_rate * global_step / warmup_steps
        elif args.lr_schedule == 'wsd':  # Warmup-Stable-Decay : plateau puis décroissance linéaire vers 0
            lr = args.learning_rate * min(1.0, (total_steps - global_step) / max(total_steps - decay_start, 1))
        else:
            lr = get_lr(global_step - warmup_steps, total_steps - warmup_steps, args.learning_rate)
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = lr * param_group.get('lr_ratio', 1.0)
        if args.ternary and args.ternary_start > 0:  # 16 bits jusqu'à ternary_start, puis rampe vers le ternaire
            lam = min(max((global_step - args.ternary_start * total_steps) / max(args.ternary_ramp * total_steps, 1), 0.0), 1.0)
            for m in ternary_layers: m.lam.fill_(lam)

        # objectif : mot à mot (ar), diffusion masquée (mdm), ou les deux en alternance d'un pas sur l'autre
        use_mdm = args.objective == 'mdm' or (args.objective == 'both' and step % 2 == 1)
        if args.objective != 'ar': set_causal(model, not use_mdm)
        res = None
        with autocast_ctx:
            if use_mdm:
                loss = mdm_loss(model, input_ids)
            else:
                res = model(input_ids, labels=labels)
                loss = res.loss + res.aux_loss
                if teacher is not None:  # distillation : imiter les probabilités de l'enseignant sur ses k meilleurs tokens
                    with torch.no_grad():
                        t_logits = teacher(input_ids).logits[:, :-1].float() / args.distill_temp
                        t_val, t_idx = t_logits.topk(args.distill_topk, dim=-1)
                        t_prob = t_val.softmax(-1)
                    s_logp = (res.logits[:, :-1].float() / args.distill_temp).log_softmax(-1).gather(-1, t_idx)
                    kl = (t_prob * (t_prob.clamp_min(1e-9).log() - s_logp)).sum(-1).mean() * args.distill_temp ** 2
                    loss = args.distill_alpha * kl + (1 - args.distill_alpha) * res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or last_step:
            for opt in optimizers: scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            for opt in optimizers: scaler.step(opt)
            scaler.update()

            for opt in optimizers: opt.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or last_step:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = 0.0 if use_mdm or res.aux_loss is None else res.aux_loss.item()
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
            track(mlflow, 'log_metrics', {"train_loss": current_loss, "learning_rate": current_lr, "tokens_seen": global_step * tokens_per_step, "tokens_per_sec": (step - start_step) * tokens_per_step / spend_time, "train_tokens_per_sec": (step - start_step) * tokens_per_step / max(spend_time - eval_seconds[0], 1e-9)}, step=global_step)

        if val_loader is not None and (step % args.eval_interval == 0 or last_step):
            eval_start = time.time()
            val_loss = evaluate(val_loader)
            eval_seconds[0] += time.time() - eval_start  # exclu du débit d'entraînement
            Logger(f'step {global_step}: val_loss {val_loss:.4f}')
            track(mlflow, 'log_metric', "val_loss", val_loss, step=global_step)
            if args.save_best and val_loss < best['val_loss'] and is_main_process():
                # garde le meilleur modèle vu en validation (protège du surapprentissage en fin de run)
                best.update(val_loss=val_loss, step=global_step)
                save_weights(f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}_best.pth")

        if args.snapshot_interval and (step % args.snapshot_interval == 0 or last_step) and is_main_process():
            save_weights(f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}_pas{global_step}.pth")

        # durée de session bientôt atteinte (machine louée ou gratuite) : point de reprise, puis arrêt propre
        time_up = args.max_hours > 0 and time.time() - run_start > args.max_hours * 3600 and step % args.accumulation_steps == 0
        if dist.is_initialized() and args.max_hours > 0:  # décision commune : celle de la carte la plus avancée
            flag = torch.tensor(float(time_up), device=args.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            time_up = bool(flag.item())
        if (step % args.save_interval == 0 or last_step or time_up) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            if args.save_resume: lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=resume_dir, optimizers=[o.state_dict() for o in optimizers], best=dict(best), mlflow_run_id=mlflow.active_run().info.run_id if mlflow else None)
            model.train()
            del state_dict

        del input_ids, labels, res, loss
        if time_up and not last_step:
            Logger(f'arrêt programmé au pas {global_step} après {args.max_hours} h : relancer avec --from_resume 1')
            stopped_early[0] = True
            return True
        if global_step >= stop_step: return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Part des steps en warmup linéaire (0 = pas de warmup)")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay AdamW (appliqué aux matrices seulement, pas aux normes)")
    parser.add_argument("--num_attention_heads", default=None, type=int, help="Nb de têtes Q (défaut config)")
    parser.add_argument("--num_key_value_heads", default=None, type=int, help="Nb de têtes K/V, GQA (défaut config)")
    parser.add_argument("--intermediate_size", default=None, type=int, help="Largeur interne du FFN (défaut config)")
    parser.add_argument("--head_dim", default=None, type=int, help="Largeur d'une tête d'attention (défaut : hidden_size / nb de têtes ; 128 pour un modèle hérité de Qwen3)")
    parser.add_argument("--max_hours", default=0.0, type=float, help="Arrêt propre après ce nombre d'heures, avec point de reprise (0 = sans limite)")
    parser.add_argument("--resume_dir", default="", type=str, help="Dossier du checkpoint de reprise (défaut : ../checkpoints) ; à placer sur un stockage qui survit à la coupure")
    parser.add_argument("--save_resume", default=1, type=int, choices=[0, 1], help="Écrire le checkpoint de reprise (modèle+optimiseur, ~750Mo) dans ../checkpoints")
    parser.add_argument("--lr_schedule", default="cosine", choices=["cosine", "wsd"], help="cosine (d'origine) ou wsd : warmup, plateau, décroissance linéaire finale")
    parser.add_argument("--decay_ratio", default=0.2, type=float, help="WSD : part finale des steps en décroissance linéaire")
    parser.add_argument("--logit_softcap", default=0.0, type=float, help="Softcap des logits (0 = désactivé, 30 = valeur usuelle)")
    parser.add_argument("--zero_init", default=0, type=int, choices=[0, 1], help="Initialise o_proj et down_proj à 0 (chaque bloc démarre comme l'identité)")
    parser.add_argument("--ffn_btt_rank", default=0, type=int, help="FFN à matrices structurées BTT de ce rang (0 = matrices pleines)")
    parser.add_argument("--ffn_btt_mid", default="128,64", type=str, help="Factorisation m1,m2 de la largeur interne du FFN structuré (m1*m2)")
    parser.add_argument("--hidden_act", default="silu", help="Activation du FFN (silu, relu2, gelu…)")
    parser.add_argument("--ffn_glu", default=1, type=int, choices=[0, 1], help="1 = FFN à porte (SwiGLU, 3 matrices) ; 0 = down(act(up(x))), 2 matrices")
    parser.add_argument("--mod_capacity", default=0.0, type=float, help="Mixture-of-Depths : part des tokens qui traversent un bloc sur deux (0 = désactivé)")
    parser.add_argument("--ffn_shared", default=0, type=int, choices=[0, 1], help="Un seul FFN partagé par tous les blocs")
    parser.add_argument("--subln", default=0, type=int, choices=[0, 1], help="RMSNorm avant o_proj et down_proj (BitNet)")
    parser.add_argument("--ternary", default="", choices=["", "matrix", "row", "group"], help="Poids ternaires −1/0/+1 dans les blocs (échelle par matrice, par ligne ou par groupe de 128)")
    parser.add_argument("--train_subset_tokens", default=0, type=int, help="Ne garder que ~N tokens d'entraînement, tirés au hasard (étude de la répétition, avec --epochs)")
    parser.add_argument("--teacher", default="", type=str, help="Poids d'un modèle ENSEIGNANT (même tokenizer) : distillation de ses probabilités")
    parser.add_argument("--teacher_hidden_size", default=768, type=int)
    parser.add_argument("--teacher_layers", default=8, type=int)
    parser.add_argument("--distill_alpha", default=0.9, type=float, help="Part de la distillation dans la loss (le reste : cross-entropy sur les vrais tokens)")
    parser.add_argument("--distill_temp", default=1.0, type=float, help="Température de la distillation")
    parser.add_argument("--distill_topk", default=50, type=int, help="Probabilités de l'enseignant limitées à ses k meilleurs tokens (Peng et al. 2024)")
    parser.add_argument("--objective", default="ar", choices=["ar", "mdm", "both"], help="ar = mot à mot ; mdm = diffusion masquée (attention bidirectionnelle) ; both = alternance")
    parser.add_argument("--ternary_start", default=0.0, type=float, help="Fraction de l'entraînement faite en 16 bits avant de passer au ternaire (0 = ternaire dès le départ)")
    parser.add_argument("--ternary_ramp", default=0.02, type=float, help="Durée (fraction des pas) de la rampe 16 bits → ternaire")
    parser.add_argument("--ternary_hadamard", default=0, type=int, choices=[0, 1], help="Rotation de Walsh-Hadamard de l'entrée des matrices ternaires")
    parser.add_argument("--tokenizer_path", default="../model", type=str, help="Dossier du tokenizer (le vocabulaire du modèle en découle)")
    parser.add_argument("--system_prompts", default="zh", choices=["zh", "fr_en"], help="SFT : liste des prompts système ajoutés au hasard (zh = liste d'origine)")
    parser.add_argument("--task", default="pretrain", choices=["pretrain", "sft"], help="pretrain = texte brut ; sft = conversations (loss sur les réponses de l'assistant)")
    parser.add_argument("--from_dir", default="../out", type=str, help="Dossier des poids de départ (--from_weight)")
    parser.add_argument("--max_steps", default=0, type=int, help="Budget en steps (0 = epochs complètes)")
    parser.add_argument("--save_best", default=0, type=int, choices=[0, 1], help="Sauver aussi le meilleur modèle en validation (*_best.pth)")
    parser.add_argument("--memory_layers", default="", type=str, help="Blocs recevant une couche mémoire, ex. '4' ou '2,5' (vide = aucune)")
    parser.add_argument("--memory_keys", default=256, type=int, help="Nb de sous-clés par moitié (table de memory_keys² lignes)")
    parser.add_argument("--memory_lr_ratio", default=1.0, type=float, help="Multiplicateur du LR AdamW pour la table et les clés de la mémoire")
    parser.add_argument("--grid_layers", default="", type=str, help="Blocs dont le FFN est remplacé par des grilles 3D, ex. '0,1,2,3,4,5,6,7' (vide = aucun)")
    parser.add_argument("--grid_count", default=16, type=int, help="Grilles 3D lues par bloc")
    parser.add_argument("--grid_features", default=8, type=int, help="Nombres par case de grille")
    parser.add_argument("--stop_fraction", default=0.0, type=float, help="Arrêt (avec évaluation finale) à cette fraction des pas, le taux d'apprentissage suivant le calendrier de l'essai complet : comparer des réglages sans finir les essais (0 = essai complet)")
    parser.add_argument("--snapshot_interval", default=0, type=int, help="Garde une copie des poids tous les N pas (<nom>_<largeur>_pas<N>.pth) : étude de la trajectoire (0 = non)")
    parser.add_argument("--ngram_orders", default="", type=str, help="Tables de n-grammes à l'entrée, ex. '2,3,4' (vide = aucune)")
    parser.add_argument("--ngram_rows", default=262144, type=int, help="Lignes de la table de chaque ordre")
    parser.add_argument("--ngram_dim", default=64, type=int, help="Largeur des lignes des tables de n-grammes")
    parser.add_argument("--detailed_eval", default=0, type=int, choices=[0, 1], help="Évaluation détaillée en fin de run (par source, par fréquence de token)")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"], help="muon = Muon pour les matrices des blocs + AdamW pour le reste")
    parser.add_argument("--muon_lr", default=0.02, type=float, help="LR de pointe de Muon (suit le même warmup/cosinus que l'AdamW)")
    parser.add_argument("--packing", default=0, type=int, choices=[0, 1], help="Sequence packing : blocs pleins sans padding (0=non, 1=oui)")
    parser.add_argument("--max_tokens", default=0, type=int, help="Budget de tokens traités (batch_size*max_seq_len par step) ; 0 = epochs complètes")
    parser.add_argument("--val_docs", default=2000, type=int, help="Nb de documents (fin du fichier) réservés à la validation ; 0 = pas de validation")
    parser.add_argument("--eval_interval", default=500, type=int, help="Intervalle (steps) d'évaluation de la val_loss")
    parser.add_argument("--use_mlflow", action="store_true", help="Logger params/métriques dans MLflow")
    parser.add_argument("--mlflow_uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.tgu.ovh"), help="URI du serveur MLflow")
    parser.add_argument("--mlflow_experiment", type=str, default="minimind-pretrain", help="Nom de l'expérience MLflow")
    parser.add_argument("--run_name", type=str, default=None, help="Nom du run MLflow")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized() and torch.cuda.is_available(): args.device = f"cuda:{local_rank}"
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    arch_overrides = {k: getattr(args, k) for k in ('num_attention_heads', 'num_key_value_heads', 'intermediate_size', 'head_dim') if getattr(args, k) is not None}
    if args.logit_softcap: arch_overrides['logit_softcap'] = args.logit_softcap
    if args.ternary: arch_overrides.update(ternary=args.ternary, ternary_hadamard=bool(args.ternary_hadamard))
    arch_overrides.update(hidden_act=args.hidden_act, ffn_glu=bool(args.ffn_glu), subln=bool(args.subln), ffn_shared=bool(args.ffn_shared), mod_capacity=args.mod_capacity)
    if args.memory_layers: arch_overrides.update(memory_layers=tuple(int(v) for v in args.memory_layers.split(',')), memory_keys=args.memory_keys)
    if args.grid_layers: arch_overrides.update(grid_layers=tuple(int(v) for v in args.grid_layers.split(',')), grid_count=args.grid_count, grid_features=args.grid_features)
    if args.ngram_orders: arch_overrides.update(ngram_orders=tuple(int(v) for v in args.ngram_orders.split(',')), ngram_rows=args.ngram_rows, ngram_dim=args.ngram_dim)
    if args.ffn_btt_rank: arch_overrides.update(ffn_btt_rank=args.ffn_btt_rank, ffn_btt_mid=tuple(int(v) for v in args.ffn_btt_mid.split(',')))
    arch_overrides['vocab_size'] = len(AutoTokenizer.from_pretrained(args.tokenizer_path))
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe), **arch_overrides)
    resume_dir = args.resume_dir or '../checkpoints'
    # un point de reprise au milieu d'un cumul de gradient perdrait les gradients déjà accumulés
    assert not args.save_resume or args.save_interval % args.accumulation_steps == 0, "--save_interval doit être un multiple de --accumulation_steps pour une reprise exacte"
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=resume_dir) if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda', dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=args.tokenizer_path, save_dir=args.from_dir, device=args.device)
    pad_id, best, sft_lengths = tokenizer.pad_token_id, {'val_loss': float('inf'), 'step': 0}, None
    # tokens déjà en cache (.npy) : inutile de relire et de convertir tout le corpus (plusieurs Go, plusieurs minutes)
    token_caches = [f'{args.data_path}.{part}_rand{args.val_docs}.npy' for part in ('train', 'val')]
    cached = args.task == 'pretrain' and args.packing and all(os.path.exists(c) for c in token_caches)
    raw_ds = load_dataset('json', data_files=args.data_path, split='train') if args.task == 'pretrain' and not cached else None
    if args.task == 'sft':
        from dataset.lm_dataset import SYSTEM_PROMPTS_FR_EN
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len, system_prompts=SYSTEM_PROMPTS_FR_EN if args.system_prompts == 'fr_en' else None)
        raw_ds = train_ds.samples
    # split train/val ALÉATOIRE (seed fixe) : le fichier est trié par source (ex. QCM anglais à la fin)
    val_idx = set() if cached else set(random.Random(0).sample(range(len(raw_ds)), args.val_docs))
    train_idx = [] if cached else [i for i in range(len(raw_ds)) if i not in val_idx]
    if args.task == 'sft':
        sft_lengths = sft_char_lengths(raw_ds, f'{args.data_path}.charlen.npy')[train_idx]
        val_ds = copy.copy(train_ds)
        val_ds.samples = raw_ds.select(sorted(val_idx))
        train_ds.samples = raw_ds.select(train_idx)
    elif args.packing:
        train_ds = PackedPretrainDataset(None if cached else raw_ds.select(train_idx), tokenizer, args.max_seq_len, cache_path=token_caches[0])
    else:
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        train_ds.samples = train_ds.samples.select(train_idx)
    val_loader = None
    if args.task == 'sft':
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_sft)
        # fichier de métadonnées (corpus/sft_sources.py) : langue de chaque conversation → loss par langue en fin de run
        meta_path = args.data_path[:-len('.jsonl')] + '.meta.jsonl'
        val_by_lang = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f: langs = [json.loads(line)['lang'] for line in f]
            for lang in sorted(set(langs)):
                ds = copy.copy(val_ds)
                ds.samples = raw_ds.select([i for i in sorted(val_idx) if langs[i] == lang])
                val_by_lang[lang] = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_sft)
    elif args.val_docs > 0:
        val_ds = PackedPretrainDataset(None if cached else raw_ds.select(sorted(val_idx)), tokenizer, args.max_seq_len, cache_path=token_caches[1])
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    if args.train_subset_tokens:
        train_ds.restrict(args.train_subset_tokens)
        Logger(f'répétition : {len(train_ds)} séquences gardées (~{len(train_ds) * args.max_seq_len / 1e6:.0f}M tokens)')
    tokens_per_step = args.batch_size * args.max_seq_len * world_size  # mode distribué : un lot par carte et par pas
    steps_per_epoch = (len(train_ds) + args.batch_size * world_size - 1) // (args.batch_size * world_size)
    total_steps = args.epochs * steps_per_epoch
    if args.max_tokens > 0: total_steps = min(total_steps, args.max_tokens // tokens_per_step)
    if args.max_steps > 0: total_steps = min(total_steps, args.max_steps)
    # lecture anticipée : même calendrier du taux d'apprentissage que l'essai complet, arrêt à une fraction du parcours
    stop_step = max(1, int(args.stop_fraction * total_steps)) if args.stop_fraction > 0 else total_steps
    Logger(f'train: {len(train_ds)} séquences, total_steps: {total_steps} (~{total_steps * tokens_per_step / 1e6:.0f}M tokens)')
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    if args.zero_init:  # sortie de chaque sous-bloc nulle au départ : h + 0 = h
        for n, p in model.named_parameters():
            if n.endswith(('o_proj.weight', 'down_proj.weight')): torch.nn.init.zeros_(p)
    # Muon : matrices des blocs (q/k/v/o, gate/up/down) ; AdamW : embedding (partagé avec lm_head) et normes
    mem_params = [p for n, p in model.named_parameters() if 'mem_' in n]  # table et clés de la mémoire : AdamW, sans decay
    muon_params = [p for n, p in model.named_parameters() if args.optimizer == 'muon' and p.dim() == 2 and 'embed_tokens' not in n and 'mem_' not in n]
    # noyaux 3D des matrices structurées (BTT) : Muon appliqué bloc par bloc (trainer/muon.py)
    core_params = [p for p in model.parameters() if args.optimizer == 'muon' and p.dim() == 3]
    muon_ids = {id(p) for p in muon_params + core_params + mem_params}
    decay = [p for p in model.parameters() if p.dim() >= 2 and id(p) not in muon_ids]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}] + ([{'params': mem_params, 'weight_decay': 0.0, 'lr_ratio': args.memory_lr_ratio}] if mem_params else []), lr=args.learning_rate)
    optimizers = [optimizer]
    if args.optimizer == 'muon':
        optimizers.append(torch.optim.Muon([{'params': muon_params, 'lr_ratio': args.muon_lr / args.learning_rate}], lr=args.muon_lr, weight_decay=args.weight_decay))
        if core_params:
            from trainer.muon import MuonAdamW
            optimizers.append(MuonAdamW([{'params': core_params, 'use_muon': True, 'lr_ratio': args.muon_lr / args.learning_rate, 'weight_decay': args.weight_decay, 'lr': args.muon_lr}]))
        Logger(f'Muon : {sum(p.numel() for p in muon_params) / 1e6:.1f}M params (+ {sum(p.numel() for p in core_params) / 1e6:.1f}M en noyaux BTT) ; AdamW : {sum(p.numel() for p in decay + no_decay) / 1e6:.1f}M params')
    
    mlflow = None
    if args.use_mlflow and is_main_process():
        # délais courts : un serveur de suivi en panne ne doit ni tuer ni figer l'entraînement
        os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '2')
        os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT', '20')
        import mlflow
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            resumed_run = ckp_data.get('mlflow_run_id') if ckp_data else None
            try: mlflow.enable_system_metrics_logging()  # charge GPU/CPU et mémoire au fil du run
            except Exception as e: Logger(f'métriques système indisponibles : {e}')
            mlflow.start_run(run_id=resumed_run) if resumed_run else mlflow.start_run(run_name=args.run_name)
            from trainer.provenance import run_tags
            mlflow.set_tags(run_tags(args.data_path, args.tokenizer_path, f'{args.data_path}.train_rand{args.val_docs}.npy'))
            if not resumed_run: mlflow.log_params({**vars(args), "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3), "total_steps": total_steps})
        except Exception as e:
            Logger(f'MLflow indisponible au démarrage ({type(e).__name__}) : run non suivi, entraînement poursuivi')
            mlflow = None

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        # un état par optimiseur (AdamW, Muon, …) ; anciens checkpoints : AdamW seul
        for opt, state in zip(optimizers, ckp_data.get('optimizers', [ckp_data['optimizer']])): opt.load_state_dict(state)
        best.update(ckp_data.get('best', {}))
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    teacher = None
    if args.teacher:  # enseignant figé, même tokenizer ; ses probabilités sont recalculées à chaque pas
        teacher = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_layers, vocab_size=lm_config.vocab_size))
        teacher.load_state_dict({k: v.float() for k, v in torch.load(args.teacher, map_location='cpu').items()})
        teacher = teacher.to(args.device).eval().requires_grad_(False)
        if args.use_compile == 1: teacher = torch.compile(teacher)
        Logger(f'distillation : enseignant {args.teacher}, alpha {args.distill_alpha}, température {args.distill_temp}, top-{args.distill_topk}')
    from model.ternary import TernaryLinear
    ternary_layers = [m for m in model.modules() if isinstance(m, TernaryLinear)]
    if ternary_layers and args.ternary_start > 0:
        for m in ternary_layers: m.lam.fill_(0.0)
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        if args.task == 'sft':  # batches triés par longueur + padding dynamique (pas de reprise au milieu d'une epoch)
            loader = DataLoader(train_ds, batch_sampler=bucketed_batches(sft_lengths, args.batch_size, args.seed + epoch), collate_fn=collate_sft, num_workers=args.num_workers, pin_memory=True)
        else:
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            done = train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            done = train_epoch(epoch, loader, len(loader), 0, wandb)
        if done: break
    if stopped_early[0]:  # pas d'évaluation finale : le run n'est pas terminé
        track(mlflow, 'end_run')
        sys.exit(0)
    if args.task == 'sft' and val_by_lang and is_main_process():
        lang_losses = {f'val_loss_{lang}': evaluate(loader) for lang, loader in val_by_lang.items()}
        Logger('loss par langue : ' + '  '.join(f'{k[9:]} {v:.4f}' for k, v in lang_losses.items()))
        track(mlflow, 'log_metrics', {f'final_{k}': v for k, v in lang_losses.items()})
    if args.detailed_eval and args.task == 'pretrain' and args.objective != 'mdm' and is_main_process():
        set_causal(model, True)
        from trainer.eval_detail import detailed_eval, format_metrics
        # corpus avec un champ `lang` (français / anglais) : loss par langue ; sinon sources du corpus d'origine
        groups = val_langs(args.data_path, args.val_docs)
        metrics = detailed_eval(getattr(model.module if isinstance(model, DistributedDataParallel) else model, '_orig_mod', model.module if isinstance(model, DistributedDataParallel) else model), args.device, data=args.data_path, doc_groups=groups)
        if ternary_layers:  # part de zéros et poids bloqués au seuil d'arrondi (« zone morte », Tequila)
            with torch.no_grad():
                stats = [(m.trits()[0] == 0).float().mean().item() for m in ternary_layers]
            metrics['ternary_zero_frac'] = sum(stats) / len(stats)
        Logger(format_metrics(metrics) + (f"\nternaire : {metrics['ternary_zero_frac']:.1%} de zéros" if ternary_layers else ''))
        track(mlflow, 'log_metrics', {f'final_{k}': v for k, v in metrics.items()})
    if torch.cuda.is_available(): track(mlflow, 'log_metric', 'final_gpu_mem_go', torch.cuda.max_memory_allocated() / 2**30)
    track(mlflow, 'end_run')
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()