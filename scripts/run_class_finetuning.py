import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict
from torch.profiler import profile, record_function, ProfilerActivity

from discovr.data.transforms.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from discovr.utils.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from discovr.data.datasets import build_dataset
from discovr.engine.engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge, merge_mean_per_class
from discovr.utils.utils import NativeScalerWithGradNormCount as NativeScaler
from discovr.utils.utils import multiple_samples_collate
import discovr.utils.utils as utils
from discovr.models import modeling_finetune


def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=False)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/videos', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--data_path_csv', default='./data/annotations/train.csv', type=str,
                        help='Path to the CSV file containing the data')
    parser.add_argument('--data_path_csv_val', default='./data/annotations/val.csv', type=str,
                        help='Path to the CSV file containing the data')
    parser.add_argument('--data_path_csv_test', default='./data/annotations/test.csv', type=str,
                        help='Path to the CSV file containing the data')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51','image_folder','DIV48','GYM99','FXS1','UBS1'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--examples', default=0, type=int)
    parser.add_argument('--finetuning_crops', default=True, type=bool)
    # linear probing
    parser.add_argument('--linear_probing', action='store_true', default=False)

    # Adding missing parameters from pretraining
    parser.add_argument('--target_type', default=None, type=str,
                        choices=['pixel', 'mlp', 'dino_v1', 'dino_v2'],
                        help='Target type for pretraining (e.g., dino_v1, dino_v2, mlp)')
    parser.add_argument('--loss_func', default=None, type=str,
                        choices=['L2', 'SWAV'],
                        help='Loss function used in pretraining (e.g., SWAV, L2)')
    parser.add_argument('--num_prototypes', default=None, type=int,
                        help='Number of prototypes for SWAV')
    parser.add_argument('--sinkhorn_iterations', default=None, type=int,
                        help='Number of iterations for Sinkhorn algorithm')
    parser.add_argument('--sinkhorn_eps', default=None, type=float,
                        help='Epsilon value for Sinkhorn algorithm')
    parser.add_argument('--tokenizer_type', default=None, type=str,
                        choices=['default', 'sparse_tube'],
                        help='Type of tokenizer used (e.g., sparse_tube)')
    parser.add_argument('--skip_dino_loss', action='store_true',
                        help='Skip DINO loss calculation')
    parser.add_argument('--decoder_depth', default=None, type=int,
                        help='Depth of decoder used in pretraining')
    parser.add_argument('--distillation_teacher', default=None, type=str, 
                        choices=['dino_s', 'dino_b', 'dino_l'],
                        help='Distillation teacher model (for DINO models)')
    parser.add_argument('--mask_type', default=None, type=str,
                        choices=['random', 'tube', 'parts', 'tube_fgbg', 'multi_local'],
                        help='Masking strategy used in pretraining')
    parser.add_argument('--mask_ratio', default=None, type=float,
                        help='Ratio of masked tokens in pretraining')
    parser.add_argument('--swav_weight', type=float, default=1.0,
                        help='Weight for SWAV loss (default: 1.0)')
    parser.add_argument('--dino_weight', type=float, default=1.0,
                        help='Weight for DINO loss (default: 1.0)')
    parser.add_argument('--turbo_weight', type=float, default=1.0,
                        help='Weight for turbo loss (default: 1.0)')
    parser.add_argument('--use_turbo_training', action='store_true',
                        help='Enable Turbo Training with selective reconstruction')
    parser.add_argument('--turbo_recon_ratio', default=0.25, type=float,
                        help='Ratio of tokens to reconstruct in Turbo Training')
    parser.add_argument('--memory_size', default=0, type=int, 
                        help='Memory size for DPC')
    parser.add_argument('--kwindow', default=1, type=int, 
                        help='Memory size for DPC')
    parser.add_argument('--use_torchcodec', action='store_true',
                        help='Use torchcodec VideoDecoder instead of decord for video loading')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = False

    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        args.data_path_csv = args.data_path_csv_val
        dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    args.data_path_csv = args.data_path_csv_test
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=True,
        prefetch_factor=2
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=2
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=2
        )
    else:
        data_loader_test = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

   
    print(f"Creating model: {args.model}")
    # Compute output dim for decoder
    if args.target_type == 'pixel':
        if args.tokenizer_type == 'sparse_tube':
            # For sparse tokenization with pixel targets, use the tokenizer's output dimension
            dec_dim = 768  # Match sparse tokenizer output dimension
        else:
            # Standard pixel targets with uniform tokenization
            dec_dim = 1536  # 16×16×3×2 = 1536
    elif 'dino' in args.target_type:
        if args.distillation_teacher == 'dino_s':
            dec_dim = 384
        elif args.distillation_teacher == 'dino_b':
            dec_dim = 768
    else:
        dec_dim = 256
        
        # Create model with necessary parameters to match pretraining architecture
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_cls=not args.use_mean_pooling,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        # Include parameters from pretraining that affect model architecture
        tokenizer_type=args.tokenizer_type,
        target_type=args.target_type,
        decoder_depth=args.decoder_depth,
        decoder_num_classes=dec_dim,
        loss_func=args.loss_func,
        num_prototypes=args.num_prototypes,
        sinkhorn_iterations=args.sinkhorn_iterations,
        eps=args.sinkhorn_eps,
        skip_dino_loss=args.skip_dino_loss,
    )
    
    if hasattr(model, 'module'):
        patch_size = model.module.patch_embed.patch_size
    else:
        patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    # args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    if args.tokenizer_type == 'sparse_tube':
            print("Sparse tube tokenizer: Using default window_size as reference only")
            # For sparse tubes, we don't use the traditional window_size calculation
            # but we still need a placeholder value for masking generator and other components
            args.window_size = (args.num_frames//2, args.input_size // 16, args.input_size // 16)
            effective_patch_size = 16  # Default reference value for sparse tubes
    else:
        # Regular window_size calculation for default tokenizer
        args.window_size = (args.num_frames // 2, args.input_size // patch_size, args.input_size // patch_size)
        args.patch_size = patch_size
        effective_patch_size = args.patch_size  # Use the configured patch_size
        
    print(f"window_size = {args.window_size}")
    # import pdb; pdb.set_trace()
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu',weights_only=False)
        # import pdb; pdb.set_trace()
        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                # import pdb; pdb.set_trace()
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        # import pdb; pdb.set_trace()
        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding 
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        loading_output = model.load_state_dict(checkpoint_model, strict=False)
        print('Loading output: ', loading_output)
        # utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix,strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                find_unused_parameters=True if args.linear_probing else False)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.linear_probing:
        print('Warning! running linear probing!\nOnly train fc layer!')
        param_require_grad = list()
        for k,v in model.named_parameters():
            if ('head' in k) or ('fc_norm' in k) or ('prototypes' in k if hasattr(model, 'prototypes') else False): 
                param_require_grad.append(k)
                v.requires_grad = True
            else:
                v.requires_grad = False
        print(f'Warning! these params require grad:\n{param_require_grad}')

    if args.eval:

        utils.auto_load_best_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    else:
        utils.auto_load_model(
              args=args, model=model, model_without_ddp=model_without_ddp,
              optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:

        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = final_test(data_loader_test, model, device, preds_file)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1,
                        'Final Top-5': final_top5}

            final_top1_per_class ,final_top5_per_class = merge_mean_per_class(args.output_dir, num_tasks,args.nb_classes)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Mean-Top-1: {final_top1_per_class:.2f}%, Mean-Top-5: {final_top5_per_class:.2f}%")
            log_stats = {'Final mean_per_class top-1': final_top1_per_class,
                        'Final mean_per_class Top-5': final_top5_per_class}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)
        

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    activities = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(args.start_epoch, args.epochs):
            with record_function("epoch"):
                with record_function("train_loop"):
                    if args.distributed:
                        data_loader_train.sampler.set_epoch(epoch)
                    if log_writer is not None:
                        log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
                    train_stats = train_one_epoch(
                        model, criterion, data_loader_train, optimizer,
                        device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                        log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                        lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                        num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
                    )
                
                with record_function("eval_loop"):
                    if data_loader_val is not None:
                        test_stats = validation_one_epoch(data_loader_val, model, device)
                        print(f"Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%")
                        if max_accuracy < test_stats["acc1"]:
                            max_accuracy = test_stats["acc1"]
                            if args.output_dir and args.save_ckpt:
                                utils.save_model(
                                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                    loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

                        print(f'Max accuracy: {max_accuracy:.2f}%')
                        if log_writer is not None:
                            log_writer.update(val_acc1=test_stats['acc1'], head="perf", step=epoch)
                            log_writer.update(val_acc5=test_stats['acc5'], head="perf", step=epoch)
                            log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)

                        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                     **{f'val_{k}': v for k, v in test_stats.items()},
                                     'epoch': epoch,
                                     'n_parameters': n_parameters}
                    else:
                        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                     'epoch': epoch,
                                     'n_parameters': n_parameters}
                    if args.output_dir and utils.is_main_process():
                        if log_writer is not None:
                            log_writer.flush()
                        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n")

            prof.step()

    # Print top 10 time-consuming operations
    print(prof.key_averages().table(
        sort_by="cpu_time_total", 
        row_limit=10))

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(data_loader_test, model, device, preds_file)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1,
                    'Final Top-5': final_top5}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
