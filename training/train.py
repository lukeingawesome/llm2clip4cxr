# training/train.py
import json
import logging
import math
import os
import time
import gc

import numpy as np
import torch
import torch.nn as nn
try:
    from torch._six import inf
except Exception:
    from torch import inf
import torch.nn.functional as F
import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None

from eva_clip import ClipLoss, get_cast_dtype, get_tokenizer
from .distributed import is_master
from .zeroshot_retrieval import retrieval_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, getattr(optimizer, "_global_grad_norm", None)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.to(dtype=torch.float32)


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


def train_one_epoch(model, tokenizer, data, epoch,
                    optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
    )

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # Re-use the encoder that is already inside the text tower
    l2v = model.text.model            # the original LLM2Vec encoder
    logit_scale = model.logit_scale

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = {k: v.to(device=device, non_blocking=True) for k, v in texts.items()}

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with autocast():
            # Vision tower
            image_features = model.visual(images)
            # Text tower (LLM2Vec + projection)
            text_features = model.text.projection(
                l2v(texts).to(dtype=cast_dtype)
            )
            total_loss, acc = loss(image_features, text_features, logit_scale)
            clip_loss = total_loss.clone().detach()

        # Optional cross-rank NaN/Inf monitoring
        if _dist_ready():
            loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
            dist.all_gather(loss_list, total_loss)
            loss_list = torch.stack(loss_list)
            loss_list_isnan = torch.isnan(loss_list).any()
            loss_list_isinf = torch.isinf(loss_list).any()
            if loss_list_isnan or loss_list_isinf:
                logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")
        else:
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logging.info(" ==================== local loss is NaN/Inf ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Clamp logit_scale as in CLIP
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for ii, v in enumerate(optimizer.param_groups):
                if v.get('group', '') == 'visual' and v.get('lr_scale', 1.0) == 1.0:
                    index_visual = ii
                if v.get('group', '') == 'text' and v.get('lr_scale', 1.0) == 1.0:
                    index_text = ii

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / max(1e-6, batch_time_m.val):#g}/s"
            )

            # loggers
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": float(grad_norm_m.val) if isinstance(grad_norm_m.val, torch.Tensor) else grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / max(1e-6, batch_time_m.val),
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # reset time meters per window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, tokenizer, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    l2v = model.text.model  # raw LLM2Vec encoder
    retrieval_zero_shot_metrics = retrieval_eval(model, l2v, data, epoch, args)
    metrics.update(retrieval_zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0
        or epoch==-1 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        logit_scale = model.logit_scale
        logit_scale_val = None

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = {k: v.to(device=device, non_blocking=True) for k, v in texts.items()}
                with autocast():
                    image_features = model.visual(images)
                    text_features = model.text.projection(
                        l2v(texts).to(dtype=cast_dtype)
                    )
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale_val = logit_scale.mean().exp()
                    logits_per_image = logit_scale_val * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / max(1, num_samples):.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale_val.cpu(),
            )
            loss = cumulative_loss / max(1, num_samples)
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def extract_features(model, data, args, device):
    img_emb_folder = args.img_emb_path
    text_emb_folder = args.text_emb_path

    save_interval = args.save_interval if args.save_interval else 100
    all_features = []
    feature_info = {}

    model.eval()
    cast_dtype = get_cast_dtype(args.precision)
    if 'val' in data:
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        all_image_features = []
        all_text_features = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                idx = i + 1

                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                image_features, text_features = model(images, texts)

                all_image_features.append(image_features)
                all_text_features.append(text_features)

                batch_size = images.shape[0]
                num_samples += batch_size
                logging.info(
                    f"Extract RANK: {args.rank} [{num_samples} / {samples_per_val}]"
                )

                if idx % save_interval == 0:
                    img_feat = np.concatenate(all_image_features)
                    text_feat = np.concatenate(all_text_features)

                    split = "%08d" % (idx // save_interval)
                    out_img_feat_file = (
                        f"{img_emb_folder}/rank{args.rank}_img_emb_{split}.npy"
                    )
                    out_text_feat_file = (
                        f"{text_emb_folder}/rank{args.rank}_text_emb_{split}.npy"
                    )

                    np.save(out_img_feat_file, img_feat)
                    np.save(out_text_feat_file, text_feat)

                    all_image_features = []
                    all_text_features = []

            if len(all_image_features) > 0:
                img_feat = np.concatenate(all_image_features)
                text_feat = np.concatenate(all_text_features)

                split = "%08d" % ((idx // save_interval) + 1)
                out_img_feat_file = (
                    f"{img_emb_folder}/rank{args.rank}_img_emb_{split}.npy"
                )
                out_text_feat_file = (
                    f"{text_emb_folder}/rank{args.rank}_text_emb_{split}.npy"
                )

                np.save(out_img_feat_file, img_feat)
                np.save(out_text_feat_file, text_feat)
    torch.distributed.barrier()
