import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import copy
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
from model.model_minimind import MiniMindConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset, PackedPretrainDataset, SFTDataset
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


def save_weights(path):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, path)


@torch.no_grad()
def evaluate(val_loader):
    random.seed(0)  # SFT : prompt système et balises tirés au hasard par le dataset → validation reproductible
    model.eval()
    losses = []
    for input_ids, labels in val_loader:
        with autocast_ctx:
            losses.append(model(input_ids.to(args.device), labels=labels.to(args.device)).loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        global_step = epoch * iters + step
        last_step = step == iters or global_step >= total_steps
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

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
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
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
            if mlflow: mlflow.log_metrics({"train_loss": current_loss, "learning_rate": current_lr, "tokens_seen": global_step * tokens_per_step, "tokens_per_sec": (step - start_step) * tokens_per_step / spend_time}, step=global_step)

        if val_loader is not None and (step % args.eval_interval == 0 or last_step):
            val_loss = evaluate(val_loader)
            Logger(f'step {global_step}: val_loss {val_loss:.4f}')
            if mlflow: mlflow.log_metric("val_loss", val_loss, step=global_step)
            if args.save_best and val_loss < best['val_loss'] and is_main_process():
                # garde le meilleur modèle vu en validation (protège du surapprentissage en fin de run)
                best.update(val_loss=val_loss, step=global_step)
                save_weights(f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}_best.pth")

        if (step % args.save_interval == 0 or last_step) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            if args.save_resume: lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss
        if global_step >= total_steps: return True
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
    parser.add_argument("--save_resume", default=1, type=int, choices=[0, 1], help="Écrire le checkpoint de reprise (modèle+optimiseur, ~750Mo) dans ../checkpoints")
    parser.add_argument("--lr_schedule", default="cosine", choices=["cosine", "wsd"], help="cosine (d'origine) ou wsd : warmup, plateau, décroissance linéaire finale")
    parser.add_argument("--decay_ratio", default=0.2, type=float, help="WSD : part finale des steps en décroissance linéaire")
    parser.add_argument("--logit_softcap", default=0.0, type=float, help="Softcap des logits (0 = désactivé, 30 = valeur usuelle)")
    parser.add_argument("--zero_init", default=0, type=int, choices=[0, 1], help="Initialise o_proj et down_proj à 0 (chaque bloc démarre comme l'identité)")
    parser.add_argument("--ffn_btt_rank", default=0, type=int, help="FFN à matrices structurées BTT de ce rang (0 = matrices pleines)")
    parser.add_argument("--ffn_btt_mid", default="128,64", type=str, help="Factorisation m1,m2 de la largeur interne du FFN structuré (m1*m2)")
    parser.add_argument("--ternary", default="", choices=["", "matrix", "row"], help="Poids ternaires −1/0/+1 dans les blocs (échelle par matrice ou par ligne)")
    parser.add_argument("--tokenizer_path", default="../model", type=str, help="Dossier du tokenizer (le vocabulaire du modèle en découle)")
    parser.add_argument("--task", default="pretrain", choices=["pretrain", "sft"], help="pretrain = texte brut ; sft = conversations (loss sur les réponses de l'assistant)")
    parser.add_argument("--from_dir", default="../out", type=str, help="Dossier des poids de départ (--from_weight)")
    parser.add_argument("--max_steps", default=0, type=int, help="Budget en steps (0 = epochs complètes)")
    parser.add_argument("--save_best", default=0, type=int, choices=[0, 1], help="Sauver aussi le meilleur modèle en validation (*_best.pth)")
    parser.add_argument("--memory_layers", default="", type=str, help="Blocs recevant une couche mémoire, ex. '4' ou '2,5' (vide = aucune)")
    parser.add_argument("--memory_keys", default=256, type=int, help="Nb de sous-clés par moitié (table de memory_keys² lignes)")
    parser.add_argument("--memory_lr_ratio", default=1.0, type=float, help="Multiplicateur du LR AdamW pour la table et les clés de la mémoire")
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
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    arch_overrides = {k: getattr(args, k) for k in ('num_attention_heads', 'num_key_value_heads', 'intermediate_size') if getattr(args, k) is not None}
    if args.logit_softcap: arch_overrides['logit_softcap'] = args.logit_softcap
    if args.ternary: arch_overrides['ternary'] = args.ternary
    if args.memory_layers: arch_overrides.update(memory_layers=tuple(int(v) for v in args.memory_layers.split(',')), memory_keys=args.memory_keys)
    if args.ffn_btt_rank: arch_overrides.update(ffn_btt_rank=args.ffn_btt_rank, ffn_btt_mid=tuple(int(v) for v in args.ffn_btt_mid.split(',')))
    arch_overrides['vocab_size'] = len(AutoTokenizer.from_pretrained(args.tokenizer_path))
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe), **arch_overrides)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
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
    raw_ds = load_dataset('json', data_files=args.data_path, split='train') if args.task == 'pretrain' else None
    if args.task == 'sft':
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        raw_ds = train_ds.samples
    # split train/val ALÉATOIRE (seed fixe) : le fichier est trié par source (ex. QCM anglais à la fin)
    val_idx = set(random.Random(0).sample(range(len(raw_ds)), args.val_docs))
    train_idx = [i for i in range(len(raw_ds)) if i not in val_idx]
    if args.task == 'sft':
        sft_lengths = sft_char_lengths(raw_ds, f'{args.data_path}.charlen.npy')[train_idx]
        val_ds = copy.copy(train_ds)
        val_ds.samples = raw_ds.select(sorted(val_idx))
        train_ds.samples = raw_ds.select(train_idx)
    elif args.packing:
        train_ds = PackedPretrainDataset(raw_ds.select(train_idx), tokenizer, args.max_seq_len, cache_path=f'{args.data_path}.train_rand{args.val_docs}.npy')
    else:
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        train_ds.samples = train_ds.samples.select(train_idx)
    val_loader = None
    if args.task == 'sft':
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_sft)
    elif args.val_docs > 0:
        val_ds = PackedPretrainDataset(raw_ds.select(sorted(val_idx)), tokenizer, args.max_seq_len, cache_path=f'{args.data_path}.val_rand{args.val_docs}.npy')
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    tokens_per_step = args.batch_size * args.max_seq_len
    steps_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    if args.max_tokens > 0: total_steps = min(total_steps, args.max_tokens // tokens_per_step)
    if args.max_steps > 0: total_steps = min(total_steps, args.max_steps)
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
        assert not args.save_resume, "--save_resume non supporté avec Muon (un seul état d'optimiseur sauvegardé)"
        optimizers.append(torch.optim.Muon([{'params': muon_params, 'lr_ratio': args.muon_lr / args.learning_rate}], lr=args.muon_lr, weight_decay=args.weight_decay))
        if core_params:
            from trainer.muon import MuonAdamW
            optimizers.append(MuonAdamW([{'params': core_params, 'use_muon': True, 'lr_ratio': args.muon_lr / args.learning_rate, 'weight_decay': args.weight_decay, 'lr': args.muon_lr}]))
        Logger(f'Muon : {sum(p.numel() for p in muon_params) / 1e6:.1f}M params (+ {sum(p.numel() for p in core_params) / 1e6:.1f}M en noyaux BTT) ; AdamW : {sum(p.numel() for p in decay + no_decay) / 1e6:.1f}M params')
    
    mlflow = None
    if args.use_mlflow and is_main_process():
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.run_name)
        mlflow.log_params({**vars(args), "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3), "total_steps": total_steps})

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
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
    if args.detailed_eval and is_main_process():
        from trainer.eval_detail import detailed_eval, format_metrics
        # corpus avec un champ `lang` (français / anglais) : loss par langue ; sinon sources du corpus d'origine
        groups = raw_ds.select(sorted(val_idx))['lang'] if 'lang' in raw_ds.column_names else None
        metrics = detailed_eval(getattr(model, '_orig_mod', model), args.device, data=args.data_path, doc_groups=groups)
        Logger(format_metrics(metrics))
        if mlflow: mlflow.log_metrics({f'final_{k}': v for k, v in metrics.items()})
    if mlflow: mlflow.end_run()
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()