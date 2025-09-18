# training/main.py
import logging
import os
import sys
import random
from datetime import datetime

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from eva_clip import (
    create_model_and_transforms,
    trace_model,
    get_tokenizer,
)

from training.data import get_data
from training.distributed import (
    is_master,
    init_distributed_device,
    world_info_from_env,
    create_deepspeed_config,
)
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import warmup_cosine_lr
from training.train import train_one_epoch, evaluate, extract_features
from training.optim import create_optimizer, get_all_parameters

# Text tower wrapper (HF)
from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def _dist_is_initialized():
    return dist.is_available() and dist.is_initialized()


def main(args):
    args, ds_init = parse_args(args)

    if ds_init is not None:
        create_deepspeed_config(args)

    if torch.cuda.is_available():
        # Enable TF32 on Ampere+ for throughput with minimal accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True

    # sanitize model name for filesystem / uri use
    args.model = args.model.replace('/', '-')

    # experiment name
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    else:
        args.name = '-'.join([
            args.name,
            datetime.now().strftime("%Y_%m_%d-%H")
        ])

    # discover world info
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)

    # logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}. '
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)

    # Build CLIP (EVA) visual + transforms
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained=None,  # Don't load pretrained weights
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_patch_dropout=args.force_patch_dropout,
        pretrained_image='',  # Don't load pretrained image weights
        pretrained_text='',   # Don't load pretrained text weights
        pretrained_visual_model=None,
        pretrained_text_model=None,
        image_mean=args.image_mean,
        image_std=args.image_std,
        cache_dir=args.cache_dir,
        skip_list=args.skip_list,
    )

    # ---------------------------------------------------------------------
    #  Text tower :  LLM2Vec (HF single-stage)  + projection
    #                (LoRA adapters added and trainable; base frozen)
    # ---------------------------------------------------------------------
    logging.info("Initialising LLM2Vec text tower from HF …")
    text_base = getattr(args, "text_base", None) or "lukeingawesome/llm2vec4cxr"
    text_model = LLM2Vec.from_pretrained(
        base_model_name_or_path=text_base,
        enable_bidirectional=True,
        pooling_mode=args.pooling_mode,     # "latent_attention" by default
        max_length=512,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    # Add LoRA adapters to the text model (if enabled)
    use_lora = getattr(args, "use_lora", True)
    if use_lora:
        lora_r = getattr(args, "lora_r", 16)
        lora_alpha = getattr(args, "lora_alpha", 32)
        lora_dropout = getattr(args, "lora_dropout", 0.1)
        force_add_lora = getattr(args, "force_add_lora", False)
        text_model.add_lora_adapters(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            force_add=force_add_lora,
            adapter_name="llm2clip",
        )
        logging.info("LoRA adapters added to text encoder")
    else:
        logging.info("LoRA adapters disabled for text encoder - training full model")
    text_model = text_model.to(dtype=torch.bfloat16)
    # tokenizer quality-of-life (optional)
    try:
        text_model.tokenizer.padding_side = 'left'
    except Exception:
        pass

    # build projection head to CLIP embed dim
    proj = nn.Sequential(
        nn.LayerNorm(text_model.config.hidden_size),
        nn.Linear(text_model.config.hidden_size, model.visual.head.out_features)
    ).to(device)

    class LLM2VecWithProj(nn.Module):
        def __init__(self, core, projection, use_lora=True):
            super().__init__()
            self.core = core
            self.projection = projection
            # compatibility with previous naming elsewhere in the codebase
            self.model = core
            self.tokenizer = core.tokenizer

            if use_lora:
                # freeze full text core except LoRA params for the *active* adapter (if any)
                lora_trainable = 0
                total = 0
                active_adapter = None
                try:
                    if hasattr(self.core, "model") and hasattr(self.core.model, "active_adapter"):
                        active_adapter = self.core.model.active_adapter
                    elif hasattr(self.core, "active_adapter"):
                        active_adapter = self.core.active_adapter
                except Exception:
                    active_adapter = None

                for n, p in self.core.named_parameters():
                    total += p.numel()
                    if ("lora_" in n) or ("lora_A" in n) or ("lora_B" in n):
                        # Train only the active adapter's LoRA tensors (names include ".{adapter}.")
                        if (active_adapter is None) or (f".{active_adapter}." in n):
                            p.requires_grad = True
                            lora_trainable += p.numel()
                        else:
                            p.requires_grad = False
                    else:
                        p.requires_grad = False
                if lora_trainable == 0:
                    logging.warning("No LoRA params detected in text core — LoRA adapters may not have been added properly. "
                                 "Projection head remains trainable.")
                else:
                    logging.info(f"Found {lora_trainable:,} LoRA parameters in text core — these will be trained. "
                               f"Base model parameters remain frozen.")
            else:
                # When LoRA is disabled, keep text encoder frozen (only projection head will be trainable)
                text_trainable = 0
                total = 0
                for n, p in self.core.named_parameters():
                    total += p.numel()
                    p.requires_grad = False  # Keep text encoder frozen
                    text_trainable += 0  # No text encoder parameters are trainable
                logging.info(f"LoRA disabled - text encoder remains frozen. Only projection head will be trained.")

        def forward(self, x):
            # Forward is rarely called directly in this repo, but keep it correct.
            return self.projection(self.core(x))

        def lock(self, *_, **__):
            pass

        def set_grad_checkpointing(self, enable=True):
            if hasattr(self.core, "gradient_checkpointing_enable"):
                if enable:
                    self.core.gradient_checkpointing_enable()
                else:
                    self.core.gradient_checkpointing_disable()

    model.text = LLM2VecWithProj(text_model, proj, use_lora=use_lora)
    tokenizer = text_model.tokenizer

    # params / logging
    tot = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Trainable params: %s (%.2f%%)", f"{train:,}", 100 * train / tot)

    if hasattr(model, 'visual'):
        total_visual_n_parameters = sum(p.numel() for p in model.visual.parameters())
        logging.info(f'number of visual params: {total_visual_n_parameters}')
    if hasattr(model, 'text'):
        total_text_n_parameters = sum(p.numel() for p in model.text.parameters())
        logging.info(f'number of text params: {total_text_n_parameters}')
    
    # Print detailed trainable parameters breakdown
    if is_master(args):
        logging.info("=" * 80)
        logging.info("DETAILED TRAINABLE PARAMETERS BREAKDOWN:")
        logging.info("=" * 80)
        
        # Visual tower parameters
        visual_trainable = sum(p.numel() for p in model.visual.parameters() if p.requires_grad)
        visual_total = sum(p.numel() for p in model.visual.parameters())
        logging.info(f"Visual Tower: {visual_trainable:,} / {visual_total:,} trainable ({100*visual_trainable/visual_total:.1f}%)")
        
        # Text tower parameters
        text_trainable = sum(p.numel() for p in model.text.parameters() if p.requires_grad)
        text_total = sum(p.numel() for p in model.text.parameters())
        logging.info(f"Text Tower: {text_trainable:,} / {text_total:,} trainable ({100*text_trainable/text_total:.1f}%)")
        
        # LoRA parameters specifically
        lora_params = sum(p.numel() for n, p in model.text.named_parameters() 
                         if p.requires_grad and (("lora_" in n) or ("lora_A" in n) or ("lora_B" in n)))
        logging.info(f"  └─ LoRA Adapters: {lora_params:,} parameters")
        
        # Text encoder parameters (should be 0 when frozen)
        text_encoder_params = sum(p.numel() for n, p in model.text.named_parameters() 
                                 if p.requires_grad and not (("lora_" in n) or ("lora_A" in n) or ("lora_B" in n)))
        logging.info(f"  └─ Text Encoder (frozen): {text_encoder_params:,} parameters")
        
        # Projection parameters
        proj_params = sum(p.numel() for p in model.text.projection.parameters() if p.requires_grad)
        logging.info(f"  └─ Projection Head: {proj_params:,} parameters")
        
        logging.info("=" * 80)
        logging.info(f"TOTAL TRAINABLE: {train:,} / {tot:,} parameters ({100*train/tot:.1f}%)")
        logging.info("=" * 80)
        
        # Pause for 10 seconds
        import time
        logging.info("Pausing for 10 seconds before starting training...")
        time.sleep(10)
        logging.info("Starting training...")

    model.to(device)
    model_without_ddp = model

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        logging.info("Lock image tower...")
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_checkpointing:
        # enable EVA checkpointing if available
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing()
        # also enable checkpointing on the text core
        if hasattr(model.text, "set_grad_checkpointing"):
            model.text.set_grad_checkpointing(True)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # Distributed
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if not args.enable_deepspeed:
            ddp_args = {}
            if args.ddp_static_graph:
                ddp_args['static_graph'] = True
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], **ddp_args
            )
            model_without_ddp = model.module

    # optimizer & scaler
    optimizer = None
    scaler = None
    if args.train_data or args.train_data_list or args.train_data_file or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        if not args.enable_deepspeed:
            scaler = GradScaler() if args.precision == "amp" else None
            optimizer = create_optimizer(args, model_without_ddp)
        else:
            scaler = None
            if args.optimizer not in ("lamb", "adamw"):
                optimizer, optimizer_params = create_optimizer(
                    args, model_without_ddp, return_params=True
                )
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
            else:
                optimizer_params = get_all_parameters(args, model)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
        if is_master(args, local=args.log_local):
            logging.info(f"num of optimizer.param_groups: {len(optimizer.param_groups)}")

    # optional resume
    start_epoch = 0
    if args.resume is not None:
        if args.enable_deepspeed:
            if os.path.exists(args.resume):
                import glob
                all_checkpoints = glob.glob(os.path.join(args.resume, 'epoch_*'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[1]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    start_epoch = latest_ckpt
                    _, client_states = model.load_checkpoint(args.resume, tag='epoch_%d' % latest_ckpt)
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {latest_ckpt})")
                else:
                    logging.info("=> no checkpoint found at '{}'".format(args.resume))
            else:
                logging.info("=> '{}' is not existing!".format(args.resume))
        else:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint["epoch"]
                    sd = checkpoint["state_dict"]
                    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items()}
                    msg = model.load_state_dict(sd, strict=False)
                    if is_master(args):
                        missing = [k for k in getattr(msg, "missing_keys", [])
                                   if not k.startswith(('visual.rope.', 'text.text_adaptor.', 'visual.head.'))]
                        if missing:
                            logging.warning("Still missing: %s", missing)
                    if optimizer is not None and "optimizer" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    if scaler is not None and 'scaler' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler'])
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
                else:
                    msg = model.load_state_dict(checkpoint, strict=False)
                    if is_master(args):
                        missing = [k for k in getattr(msg, "missing_keys", [])
                                   if not k.startswith(('visual.rope.', 'text.text_adaptor.', 'visual.head.'))]
                        if missing:
                            logging.warning("Still missing: %s", missing)
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # data
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=tokenizer)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # scheduler
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        if is_master(args):
            logging.info(f"total_steps: {total_steps}")
        scheduler = warmup_cosine_lr(optimizer, args, total_steps)

    # logging writers
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
            settings=wandb.Settings(start_method="fork"),
        )
        if args.debug:
            wandb.watch(model, log='all')
        params_file = os.path.join(args.logs, args.name, "params.txt")
        if os.path.exists(params_file):
            wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if args.extract_features:
        with torch.no_grad():
            extract_features(model, data, args, device)
        return

    if 'train' not in data:
        evaluate(model, tokenizer, data, start_epoch, args, writer)
        return

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        train_one_epoch(model, tokenizer, data, epoch,
                        optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, tokenizer, data, completed_epoch, args, writer)

        # checkpoints
        if args.logs and args.logs.lower() != 'none' and args.enable_deepspeed:
            deepspeed_checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
            if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                ):
                client_state = {'epoch': completed_epoch}
                model.save_checkpoint(save_dir=deepspeed_checkpoint_path, tag=f"epoch_{completed_epoch}", client_state=client_state)

        elif args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
