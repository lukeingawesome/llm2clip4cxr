#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrieval evaluation for LLM2CLIP (training-matched):

• EVA-CLIP visual tower
• LLM2Vec text core (HF) with the SAME LoRA shell (adapter_name) as training
• Projection head identical to training
• Robust load order:
  - from HF: full text backbone
  - from ckpt: visual.*, text.projection.*, logit_scale, and LoRA deltas (text.model.* / text.core.*)
• Similarity:
  - clip  : logit_scale * (I @ T^T)  [NO L2 normalization]  ← matches training
  - cosine: L2-normalized dot
  - dot   : raw dot
• If anchors==candidates, encode with ONE DataLoader (collate-aligned), like in training.
• Compute ONE similarity matrix S and read both directions from it (rows: I→T, columns: T→I).
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, List
import re

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def timeit(msg: str):
    t0 = time.time()
    def _done(extra: str = ""):
        dt = time.time() - t0
        logging.info("%s done in %.2fs%s", msg, dt, f" ({extra})" if extra else "")
        return dt
    return _done


def is_mp_rank_file(path: str) -> bool:
    name = os.path.basename(path)
    return "mp_rank_" in name and name.endswith((".pt", ".pth", ".bin"))


def find_consolidated_in_dir(d: Path) -> Optional[str]:
    for name in ("pytorch_model.bin", "consolidated.pth", "consolidated.pt", "weights.pth", "weights.pt", "model.pt"):
        p = d / name
        if p.exists():
            return str(p)
    for pat in ("*.bin", "*.pt", "*.pth"):
        for p in sorted(d.glob(pat)):
            if not is_mp_rank_file(str(p)):
                return str(p)
    return None


def resolve_ckpt(path: str) -> str:
    p = Path(path)
    if p.is_file():
        if is_mp_rank_file(str(p)):
            raise RuntimeError(
                f"Refusing DeepSpeed shard: {p.name}\n"
                f"Consolidate first:\n"
                f"  python -m deepspeed.utils.zero_to_fp32 {p.parent} {p.parent / 'pytorch_model.bin'}"
            )
        return str(p)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    cons = find_consolidated_in_dir(p)
    if cons:
        return cons
    shards = list(p.glob("mp_rank_*_model_states.*"))
    if shards:
        raise RuntimeError(
            f"Found mp_rank shards in {p} but no consolidated file.\n"
            f"Consolidate with:\n  python -m deepspeed.utils.zero_to_fp32 {p} {p / 'pytorch_model.bin'}"
        )
    raise FileNotFoundError(f"No model file found in: {path}")


def torch_load_trusted(path: str, map_location="cpu"):
    import torch
    size_mb = (os.path.getsize(path) / (1024 * 1024.0)) if os.path.exists(path) else -1
    logging.info("Loading checkpoint: %s (%.1f MB)", path, size_mb)
    tdone = timeit("torch.load")
    obj = torch.load(path, map_location=map_location)  # weights_only=False for compatibility
    tdone()
    return obj


def extract_state_dict(obj) -> Dict[str, "torch.Tensor"]:
    if isinstance(obj, dict):
        for k in ("state_dict", "module", "model_state_dict", "model", "sd", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    return obj


def count_lora_keys(sd: Dict[str, "torch.Tensor"]) -> int:
    return sum(1 for k in sd.keys() if ("lora_" in k) or (".lora_A" in k) or (".lora_B" in k))


def _is_text_core_lora_key(subkey: str) -> bool:
    return ("lora_" in subkey) or (".lora_A" in subkey) or (".lora_B" in subkey)


def filter_clip_sd_for_eval(sd: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    """
    Keep ONLY visual.*, text.projection.*, logit_scale, and LoRA deltas in text core.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("visual.") or k.startswith("text.projection.") or k == "logit_scale":
            out[k] = v
            continue
        if k.startswith("text.model."):
            sub = k.split("text.model.", 1)[1]
            if _is_text_core_lora_key(sub):
                out[k] = v
        elif k.startswith("text.core."):
            sub = k.split("text.core.", 1)[1]
            if _is_text_core_lora_key(sub):
                out[k] = v
    return out


def detect_lora_adapter_names_in_keys(keys) -> set:
    pat = re.compile(r"\.lora_[AB]\.([^.]+)\.weight$")
    names = set()
    for k in keys:
        m = pat.search(k)
        if m:
            names.add(m.group(1))
    return names


def remap_lora_adapter_names(sd: Dict[str, "torch.Tensor"], target: str):
    """
    Rename LoRA keys with adapter suffix to the `target` adapter name if needed.
    E.g. ...lora_A.default.weight -> ...lora_A.llm2clip.weight
    """
    if not target:
        return sd, 0, None
    pat = re.compile(r"(.*\.lora_[AB]\.)([^.]+)(\.weight)$")
    out, renamed, src = {}, 0, None
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            cur = m.group(2)
            if cur != target:
                if src is None:
                    src = cur
                k2 = m.group(1) + target + m.group(3)
                out[k2] = v
                renamed += 1
            else:
                out[k] = v
        else:
            out[k] = v
    return out, renamed, src


def parse_modules_arg(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return [t.strip() for t in s.split(",") if t.strip()]


def parse_args():
    p = argparse.ArgumentParser("LLM2CLIP retrieval (training-matched, collate-aligned)")

    # I/O
    p.add_argument("--anchors", required=True, help="CSV for image queries (image→text)")
    p.add_argument("--candidates", required=True, help="CSV for text queries (text→image)")
    p.add_argument("--save", default="", help="Where to save CSV of ranks + JSON metrics")
    p.add_argument("--exit-no-segfault", action="store_true")

    # Checkpoint(s)
    p.add_argument("--clip-ckpt", required=False, help="Consolidated CLIP file or dir (training checkpoint)")
    p.add_argument("--text-base", default="lukeingawesome/llm2vec4cxr",
                   help="HF id/path for the full text core (kept as backbone)")

    # Model / precision
    p.add_argument("--model", default="EVA02-CLIP-L-14-336")
    p.add_argument("--pretrained", default="eva_clip")
    p.add_argument("--pretrained-image", default="", help="Load pretrained image model weights")
    p.add_argument("--pretrained-text", default="", help="Load pretrained text model weights")
    p.add_argument("--pretrained-visual-model", default=None, help="Pretrained visual model name")
    p.add_argument("--pretrained-text-model", default=None, help="Pretrained text model name")
    p.add_argument("--precision", default="bf16",
                   choices=["fp32", "fp16", "bf16", "bfloat16", "half", "amp"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--pooling-mode", default="latent_attention")
    p.add_argument("--text-max-len", type=int, default=512)

    # CSV schema (must match your training)
    p.add_argument("--csv-img-key", default="img_path")
    p.add_argument("--csv-caption-key", default="caption_if")

    # LoRA shell (must match training)
    p.add_argument("--use-lora", default=True, action='store_true', help="Use LoRA adapters for text encoder")
    p.add_argument("--no-lora", dest="use_lora", action='store_false', help="Disable LoRA adapters for text encoder")
    p.add_argument("--adapter-name", default="llm2clip")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # Eval options
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--similarity", choices=["clip", "cosine", "dot"], default="clip")
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("default")

    import torch
    import pandas as pd

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; switching to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # precision helpers
    from eva_clip import get_cast_dtype, create_model_and_transforms
    from training.precision import get_autocast

    cast_dtype = get_cast_dtype(args.precision)
    autocast_ctx = get_autocast(args.precision)

    # 1) Visual backbone (+ transforms)
    logging.info("Step 1: build EVA-CLIP visual …")
    tdone = timeit("create_model_and_transforms")
    model, _, preprocess_val = create_model_and_transforms(
        args.model, args.pretrained,  # Allow pretrained weights if specified
        precision=args.precision, device=device,
        jit=False, force_quick_gelu=False,
        force_patch_dropout=False,
        pretrained_image=args.pretrained_image, pretrained_text=args.pretrained_text,
        pretrained_visual_model=args.pretrained_visual_model, pretrained_text_model=args.pretrained_text_model,
        image_mean=None, image_std=None, cache_dir=None, skip_list=None,
    )
    tdone()

    # 2) Text tower (LLM2Vec core) with projection head as in training
    logging.info("Step 2: build text tower (LLM2Vec core) …")
    try:
        from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
    except Exception:
        from training.llm2vec_wrapper import LLM2VecWrapper as LLM2Vec

    text_model = LLM2Vec.from_pretrained(
        base_model_name_or_path=args.text_base,
        enable_bidirectional=True,
        pooling_mode=args.pooling_mode,
        max_length=(args.text_max_len or None),
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    # LoRA shell (same as training)
    use_lora = getattr(args, "use_lora", True)
    if use_lora:
        def parse_modules_arg(s: str) -> List[str]:
            s = (s or "").strip()
            if not s:
                return ["q_proj", "k_proj", "v_proj", "o_proj"]
            return [t.strip() for t in s.split(",") if t.strip()]
        target_modules = parse_modules_arg(getattr(args, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj"))

        text_model.add_lora_adapters(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            force_add=True,
            adapter_name=args.adapter_name,
        )
        logging.info("LoRA adapters added to text encoder for evaluation")
    else:
        logging.info("LoRA adapters disabled for text encoder evaluation")
    text_model = text_model.to(dtype=torch.bfloat16)

    try:
        text_model.tokenizer.padding_side = 'left'
    except Exception:
        pass

    proj = torch.nn.Sequential(
        torch.nn.LayerNorm(text_model.config.hidden_size),
        torch.nn.Linear(text_model.config.hidden_size, model.visual.head.out_features),
    )

    class TextWithProj(torch.nn.Module):
        def __init__(self, core, projection):
            super().__init__()
            self.core = core
            self.projection = projection
            self.model = core
            self.tokenizer = getattr(core, "tokenizer", None)

        def forward(self, enc):
            return self.projection(self.core(enc))

    model.text = TextWithProj(text_model, proj)
    model.to(device).eval()
    model.text.model.to(device)
    model.text.projection.to(device, dtype=cast_dtype)

    # 3) Load CLIP checkpoint (filtered) + adapter remap if needed
    if args.clip_ckpt:
        logging.info("Step 3: load CLIP checkpoint …")
        clip_path = resolve_ckpt(args.clip_ckpt)
        logging.info("Resolved CLIP ckpt: %s", clip_path)
        ckpt = torch_load_trusted(clip_path, map_location="cpu")
        clip_sd_raw = extract_state_dict(ckpt)
        clip_sd = filter_clip_sd_for_eval(clip_sd_raw)

        adapters_in_ckpt = detect_lora_adapter_names_in_keys(clip_sd.keys())
        if adapters_in_ckpt and (args.adapter_name not in adapters_in_ckpt):
            clip_sd, renamed, src = remap_lora_adapter_names(clip_sd, args.adapter_name)
            logging.info("Remapped %d LoRA tensors from adapter '%s' -> '%s'.",
                         renamed, src, args.adapter_name)

        # quick counts for sanity
        num_proj = sum(1 for k in clip_sd if k.startswith("text.projection."))
        num_visual = sum(1 for k in clip_sd if k.startswith("visual."))
        has_logit = "logit_scale" in clip_sd
        logging.info("CKPT filtered keys — visual: %d, text.projection: %d, logit_scale: %s",
                     num_visual, num_proj, str(has_logit))

        msg = model.load_state_dict(clip_sd, strict=False)
        logging.info("[CLIP] load_state_dict → missing=%d unexpected=%d",
                     len(getattr(msg, "missing_keys", [])), len(getattr(msg, "unexpected_keys", [])))
    else:
        logging.info("Step 3: Skipping CLIP checkpoint loading (no --clip-ckpt provided)")
        clip_sd = {}  # Initialize empty state dict when no checkpoint is loaded

    # dtype hygiene for text core
    for p in model.text.model.parameters():
        if p.dtype != torch.bfloat16:
            p.data = p.data.to(dtype=torch.bfloat16, device=device)

    n_lora = count_lora_keys(clip_sd)
    active_adapter = getattr(getattr(text_model, "model", text_model), "active_adapter", None)
    logging.info("Text core source: HF (%s) | #LoRA keys loaded from CLIP ckpt: %d | active_adapter=%s",
                 args.text_base, n_lora, str(active_adapter))
    if n_lora == 0:
        logging.warning("No LoRA tensors found in the CLIP checkpoint after filtering.")

    # 4) Datasets / loaders (collate-aligned when same CSV)
    logging.info("Step 4: build CSV datasets …")
    from training.data import CustomCSVDataset
    from torch.utils.data import DataLoader

    dsA = CustomCSVDataset(
        csv_file=args.anchors,
        transform=preprocess_val,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=model.text.tokenizer,
        is_train=False,
    )
    dsB = CustomCSVDataset(
        csv_file=args.candidates,
        transform=preprocess_val,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=model.text.tokenizer,
        is_train=False,
    )

    dfA = dsA.data_frame
    dfB = dsB.data_frame

    def _same_dataset(df1, df2) -> bool:
        if len(df1) != len(df2):
            return False
        try:
            a_img = list(df1[args.csv_img_key].astype(str))
            b_img = list(df2[args.csv_img_key].astype(str))
            a_cap = list(df1[args.csv_caption_key].astype(str))
            b_cap = list(df2[args.csv_caption_key].astype(str))
            return (a_img == b_img) and (a_cap == b_cap)
        except Exception:
            return False

    same_csv = _same_dataset(dfA, dfB)
    if same_csv:
        logging.info("Detected SAME dataset for anchors and candidates — using a SINGLE DataLoader (collate-aligned).")
    else:
        logging.info("Detected DIFFERENT datasets — encoding images and texts with separate loaders.")

    dlA = DataLoader(dsA, batch_size=args.batch, shuffle=False,
                     num_workers=args.workers, pin_memory=True, drop_last=False,
                     collate_fn=dsA.collate_fn)
    if not same_csv:
        dlB = DataLoader(dsB, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True, drop_last=False,
                         collate_fn=dsB.collate_fn)
    else:
        dlB = None

    # 5) Encode features (collate-aligned when possible)
    logging.info("Step 5: encode features …")
    import torch
    feats_i, feats_t = [], []

    def encode_images(imgs):
        with autocast_ctx():
            return model.visual(imgs)

    with torch.no_grad():
        if same_csv:
            tdone = timeit("encode (images+texts via single loader)")
            l2v = model.text.model.to(device)
            for images, texts in dlA:
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                if isinstance(texts, dict):
                    texts = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in texts.items()}
                elif hasattr(texts, "to"):
                    texts = texts.to(device=device)

                img_f = encode_images(images)
                with autocast_ctx():
                    txt_rep = l2v(texts)
                    txt_f = model.text.projection(txt_rep.to(dtype=cast_dtype))

                feats_i.append(img_f.detach().float().cpu())
                feats_t.append(txt_f.detach().float().cpu())
            tdone()
        else:
            tdone = timeit("encode images")
            for images, _ in dlA:
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                img_f = encode_images(images)
                feats_i.append(img_f.detach().float().cpu())
            tdone()

            tdone = timeit("encode texts (via collate)")
            l2v = model.text.model.to(device)
            for _, texts in dlB:
                if isinstance(texts, dict):
                    texts = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in texts.items()}
                elif hasattr(texts, "to"):
                    texts = texts.to(device=device)
                with autocast_ctx():
                    txt_rep = l2v(texts)
                    txt_f = model.text.projection(txt_rep.to(dtype=cast_dtype))
                feats_t.append(txt_f.detach().float().cpu())
            tdone()

    img_feats = torch.cat(feats_i, 0).float()
    txt_feats = torch.cat(feats_t, 0).float()

    N = min(img_feats.shape[0], txt_feats.shape[0])

    # diagnostics: norms (pre-normalization)
    img_norm = img_feats.norm(dim=1).mean().item()
    txt_norm = txt_feats.norm(dim=1).mean().item()
    logging.info("Feature norms (mean): image=%.3f | text=%.3f | N=%d", img_norm, txt_norm, N)

    # 6) Build ONE similarity matrix S and compute both directions from it
    logging.info("Step 6: compute recalls (single similarity matrix) …")

    # For training-matched 'clip', DO NOT L2-normalize. For 'cosine', do.
    if args.similarity == "cosine":
        img_feats = torch.nn.functional.normalize(img_feats, dim=1)
        txt_feats = torch.nn.functional.normalize(txt_feats, dim=1)
        logging.info("Applied L2 normalization for 'cosine' similarity.")
    elif args.similarity == "clip":
        logging.info("Using CLIP logits: logit_scale * (I @ T^T) — no L2 normalization (matches training).")
    else:
        logging.info("Using raw dot-product similarity (no normalization).")

    # Move to device for matmul, compute, then return to CPU as float32
    img_d = img_feats[:N].to(device=device, dtype=torch.float32, non_blocking=True)
    txt_d = txt_feats[:N].to(device=device, dtype=torch.float32, non_blocking=True)

    # Optional logit scale for 'clip'
    logit_scale = None
    if args.similarity == "clip":
        try:
            logit_scale = model.logit_scale.exp().item()
        except Exception:
            logging.warning("No logit_scale in model; falling back to cosine style.")
            img_d = torch.nn.functional.normalize(img_d, dim=1)
            txt_d = torch.nn.functional.normalize(txt_d, dim=1)

    with torch.no_grad():
        S = img_d @ txt_d.T
        if (args.similarity == "clip") and (logit_scale is not None):
            S = S * logit_scale
    S = S.detach().cpu().float()  # NxN

    # Sanity: diag vs off-diag means
    diag_vals = S.diag()
    offd_mean = (S.sum() - diag_vals.sum()) / (S.numel() - N)
    logging.info("Sanity sims: mean(diagonal)=%.4f | mean(off-diagonal)=%.4f",
                 diag_vals.mean().item(), offd_mean.item())

    # Compute DIAGONAL recalls from the same S
    # i2t (rows): query i ranks row i across columns
    # t2i (cols): query i ranks column i across rows
    max_k = max(args.topk)
    # i2t
    row_topk = S.topk(k=max_k, dim=1).indices  # [N, max_k]
    gt_row = torch.arange(N).unsqueeze(1)       # [N, 1]
    hits_i2t = {k: (row_topk[:, :k] == gt_row).any(dim=1).float().mean().item() for k in sorted(args.topk)}
    # t2i
    col_topk = S.topk(k=max_k, dim=0).indices  # [max_k, N], rows (images) selected per column (text)
    gt_col = torch.arange(N).unsqueeze(0)       # [1, N]
    hits_t2i = {k: (col_topk[:k, :] == gt_col).any(dim=0).float().mean().item() for k in sorted(args.topk)}

    def fmt(hits): return ", ".join([f"R@{k}={hits[k]*100:.2f}%" for k in sorted(hits.keys())])
    logging.info("DIAG — Image→Report: %s", fmt(hits_i2t))
    logging.info("DIAG — Report→Image: %s", fmt(hits_t2i))
    print("\n=== Retrieval @K (DIAGONAL, training-like) ===")
    print("Image→Report:", fmt(hits_i2t))
    print("Report→Image:", fmt(hits_t2i))

    # 7) Save artifacts
    if args.save:
        import pandas as pd
        os.makedirs(os.path.dirname(args.save), exist_ok=True)

        # ranks (optional; row ranks only, like i2t). For completeness you can add col ranks too.
        # rank_i = 1 + (S[i,:] > S[i,i]).sum()
        row_ranks = (S > diag_vals.unsqueeze(1)).sum(dim=1).to(torch.long) + 1
        
        # Get top-1 predicted indices for image->text retrieval
        top1_pred_indices = S.topk(k=1, dim=1).indices.squeeze(1).numpy()  # [N]
        
        # Get ground truth and predicted texts
        ground_truth_texts = []
        predicted_texts = []
        
        # For diagonal evaluation, we need to understand the matrix structure:
        # S[i,j] = similarity(image_i, text_j)
        # When same_csv=True: both images and texts come from dsA
        # When same_csv=False: images come from dsA, texts come from dsB
        
        # For ground truth (diagonal): we want text_i for image_i
        # For predictions (top-1): we want text_j where j is the predicted index
        
        if not same_csv:
            logging.warning("Different datasets detected. Diagonal evaluation assumes dsA[i] corresponds to dsB[i]. "
                           "Verify that your datasets are properly aligned for meaningful diagonal evaluation.")
        
        for i in range(N):
            # Ground truth text: for image i, the correct text is text i
            if same_csv:
                # Both from dsA
                gt_text = dsA.data_frame.iloc[i][args.csv_caption_key]
            else:
                # Image from dsA, text from dsB - but for diagonal, we need matching pairs
                # This assumes dsA[i] corresponds to dsB[i] for diagonal evaluation
                gt_text = dsB.data_frame.iloc[i][args.csv_caption_key]
            ground_truth_texts.append(gt_text)
            
            # Predicted text: top-1 from similarity matrix
            pred_idx = top1_pred_indices[i]
            if same_csv:
                # Both from dsA
                pred_text = dsA.data_frame.iloc[pred_idx][args.csv_caption_key]
            else:
                # Image from dsA, text from dsB
                pred_text = dsB.data_frame.iloc[pred_idx][args.csv_caption_key]
            predicted_texts.append(pred_text)

        df_out = pd.DataFrame({
            "index": list(range(N)),
            "rank_correct_report_diag": row_ranks.numpy(),
            "ground_truth_text": ground_truth_texts,
            "predicted_text": predicted_texts,
            "predicted_index": top1_pred_indices,
        })
        
        # Add some debugging info for verification
        if args.debug:
            logging.info("Sample alignment check (first 3 rows):")
            for i in range(min(3, N)):
                logging.info(f"Row {i}: GT='{ground_truth_texts[i][:50]}...' | Pred='{predicted_texts[i][:50]}...' | Pred_idx={top1_pred_indices[i]}")
                if same_csv and i == top1_pred_indices[i]:
                    logging.info(f"  -> Perfect match (diagonal)")
                elif not same_csv:
                    logging.info(f"  -> Cross-dataset prediction")
        for k in sorted(args.topk):
            df_out[f"hit@{k}_diag"] = (df_out["rank_correct_report_diag"] <= k)
        df_out.to_csv(args.save, index=False)

        # metrics json
        payload = {
            "diag": {
                "image_to_report": {f"R@{k}": float(hits_i2t[k]) for k in sorted(args.topk)},
                "report_to_image": {f"R@{k}": float(hits_t2i[k]) for k in sorted(args.topk)},
            },
            "num_queries": int(N),
            "similarity": args.similarity,
            "text_base": args.text_base,
            "clip_ckpt": args.clip_ckpt or "none",
            "adapter_name": args.adapter_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": parse_modules_arg(getattr(args, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj")),
        }
        with open(os.path.splitext(args.save)[0] + "_metrics.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {args.save}")

    if args.exit_no_segfault:
        os._exit(0)


if __name__ == "__main__":
    main()
