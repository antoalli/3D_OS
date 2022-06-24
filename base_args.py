import argparse
import ast


def add_base_args(parser):
    parser.add_argument("--local_rank",
                        type=int, help="GPU worker rank for distributed training")
    parser.add_argument("--use_sync_bn", "-sync_bn",
                        action='store_true', help="enables synchronized batchnorm")
    parser.add_argument("--use_amp", "-amp",
                        action='store_true', help="enables automatic mixed precision")
    parser.add_argument("--script_mode", "-mode",
                        type=str, default="train", choices=['train', 'train_exposure', 'eval', 'eval_rw'])
    parser.add_argument("--config",
                        type=str, help="experiment yaml configuration file")
    parser.add_argument("--seed",
                        type=int, default=1)
    parser.add_argument("--epochs",
                        default=250, type=int)
    parser.add_argument("--batch_size",
                        default=64, type=int, help="batch size for each GPU worker")
    parser.add_argument("--num_workers",
                        default=6, type=int)
    parser.add_argument("--resume",
                        type=str, default=None, help="resume training from checkpoint")
    parser.add_argument("--apply_fix_cellphone", 
                        default="True", type=ast.literal_eval, help="Enable (default) or disable fix on cellphone class")
    parser.add_argument("--data_root",
                        default="./3D_OS_release_data", help="path to datasets root dir - this folder contains all datasets")
    parser.add_argument("--checkpoints_dir",
                        type=str, default="outputs", help="root for exp files")
    parser.add_argument("--exp_name",
                        type=str, default=None, help="exp name")
    parser.add_argument("--eval_step",
                        default=1, type=int)
    parser.add_argument("--save_step",
                        default=10, type=int)
    parser.add_argument("--ckpt_path",
                        default=None, type=str, help="Ckpt path for eval")

    # Category set choice
    parser.add_argument("--src",
                        type=str, default="SN1", help="category choice",
                        choices=["SN1", "SN2", "SN3", "SR1", "SR2"])

    # parameters ScanObjectNN test
    parser.add_argument("--sonn_split",
                        default="main_split", type=str, help="scanobject data split")
    parser.add_argument("--sonn_h5_name",
                        default="objectdataset.h5", type=str, help="scanobject data h5 name")

    return parser
