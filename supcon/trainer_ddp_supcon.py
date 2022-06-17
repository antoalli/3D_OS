import sys
import os

sys.path.append(os.getcwd())

import os.path as osp
import time
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms

from models.simclr import SimCLR  # base model for contrastive learning
from supcon.contrastive_loss import simclr_loss_func
from utils.data_utils import *
from datasets.sncore_4k import ShapeNetCore4k
# noinspection PyUnresolvedReferences
from datasets.sncore_splits import *
from utils.utils import *
from utils.dist import *
from utils.ood_utils import get_sncore_id_ood_loaders, compute_centroids, compute_clf_centroids, \
    compute_sim_centroids, eval_ood_sncore_csi, eval_ood_sncore, get_ood_metrics, get_simclr_proj
import wandb
from base_args import add_base_args

"""
ShapenetCore4k contrastive learning trainer
"""


def get_args():
    parser = argparse.ArgumentParser("OOD on point clouds via contrastive learning")
    parser = add_base_args(parser)
    # experiment specific args
    parser.add_argument("--augm_set",
                        type=str, default="st", help="data augm - st is only scale+translate", choices=["all", "st"])
    parser.add_argument('--simclr', action='store_true',
                        help='self-supervised contrastive objective instead of default supcon')
    parser.add_argument("--grad_norm_clip",
                        default=-1, type=float, help="gradient clipping")
    parser.add_argument("--num_points",
                        default=1024, type=int, help="number of points sampled for each object view")
    parser.add_argument("--temp",
                        default=0.1, type=float, help="temperature for contrastive learning")
    parser.add_argument("--wandb_name",
                        type=str, default=None)
    parser.add_argument("--wandb_group",
                        type=str, default=None)
    parser.add_argument("--wandb_proj",
                        type=str, default="benchmark-3d-ood-contrastive")

    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)

    accepted_src = ["SN1", "SN2", "SN3"]
    assert args.src in accepted_src, f"Chosen class set {args.src} is not correct"
    accepted_src.remove(args.src)
    args.tar1 = accepted_src[0]
    args.tar2 = accepted_src[1]

    if args.script_mode == 'eval':
        args.batch_size = 1

    return args


def get_train_loader_dpp(opt):
    world_size = get_ws()
    rank = get_rank()
    drop_last = not str(opt.script_mode).startswith('eval')
    if opt.augm_set == 'all':
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),
            AugmScale(lo=2 / 3, hi=3 / 2),
            AugmRotate(axis=[0.0, 1.0, 0.0]),
            AugmRotatePerturbation(),
            AugmTranslate(translate_range=0.2),
            AugmJitter()]
    elif opt.augm_set == 'st':
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),
            AugmScale(lo=2 / 3, hi=3 / 2),
            AugmTranslate(translate_range=0.2)]
    else:
        raise ValueError(f"Unknown augmentation set: {opt.augm_set}")

    print(f"Train transforms: {set_transforms}")
    train_transforms = TwoCropTransform(transforms.Compose(set_transforms))
    train_data = ShapeNetCore4k(
        data_root=opt.data_root,
        split="train",
        class_choice=list(eval(opt.src).keys()),
        num_points=4096,  # sampling to num_points as data augm
        transforms=train_transforms)

    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers, sampler=train_sampler,
        worker_init_fn=init_np_seed)

    return train_loader


def get_test_loaders_ddp(opt):
    """
    DataLoaders used for testing / ood evaluation
    Compatible with DDP training runtime evaluation

    Args:
        opt: script arguments

    Returns:
        src: name of ID category set
        targets: list of names for tar1 and tar2
        train_loader: train loader to compute category centroids - no augm, shuffle=True, drop_last=True
        test_loader: ID data loader - no augm, shuffle=True, drop_last=True if Training else False
        tar1_loader: OOD 1 data loader - no augm, shuffle=True, drop_last=True if Training else False
        tar2_loader: OOD 2 data loader - no augm, shuffle=True, drop_last=True if Training else False

    """

    ws = get_ws()
    rank = get_rank()
    drop_last = not str(opt.script_mode).startswith('eval')
    src = opt.src
    targets = [opt.tar1, opt.tar2]
    print(f"OOD src: {src}, targets: {targets}")

    # source
    train_data = ShapeNetCore4k(
        data_root=opt.data_root,
        split="train",
        class_choice=list(eval(src).keys()),
        num_points=opt.num_points,
        transforms=None)

    test_data = ShapeNetCore4k(
        data_root=opt.data_root,
        split="test",
        class_choice=list(eval(src).keys()),
        num_points=opt.num_points,
        transforms=None)

    # targets
    tar1_data = ShapeNetCore4k(
        data_root=opt.data_root,
        split="test",
        class_choice=list(eval(targets[0]).keys()),
        num_points=opt.num_points,
        transforms=None)

    tar2_data = ShapeNetCore4k(
        data_root=opt.data_root,
        split="test",
        class_choice=list(eval(targets[1]).keys()),
        num_points=opt.num_points,
        transforms=None)

    train_sampler = DistributedSampler(train_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    test_sampler = DistributedSampler(test_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    tar1_sampler = DistributedSampler(tar1_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    tar2_sampler = DistributedSampler(tar2_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None

    train_loader = DataLoader(train_data,
                              batch_size=opt.batch_size,
                              drop_last=drop_last,
                              num_workers=int(opt.num_workers // 2),
                              sampler=train_sampler,
                              worker_init_fn=init_np_seed)
    test_loader = DataLoader(test_data,
                             batch_size=opt.batch_size,
                             drop_last=drop_last,
                             num_workers=(opt.num_workers // 2),
                             sampler=test_sampler,
                             worker_init_fn=init_np_seed)
    tar1_loader = DataLoader(tar1_data,
                             batch_size=opt.batch_size,
                             drop_last=drop_last,
                             num_workers=(opt.num_workers // 2),
                             sampler=tar1_sampler,
                             worker_init_fn=init_np_seed)
    tar2_loader = DataLoader(tar2_data,
                             batch_size=opt.batch_size,
                             drop_last=drop_last,
                             num_workers=(opt.num_workers // 2),
                             sampler=tar2_sampler,
                             worker_init_fn=init_np_seed)

    return src, targets, train_loader, test_loader, tar1_loader, tar2_loader


def train_epoch_supcon(epoch, args, loader, model, scaler, optimizer, io_logger):
    """
    Train one epoch
    model: encoder + projector
    loss: supervised contrastive
    Projection normalization is performed in loss function
    """
    rank = get_rank()
    ws = get_ws()
    time1 = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    model.train()

    for i, batch in enumerate(loader, 0):
        optimizer.zero_grad(set_to_none=True)
        args.glob_it += 1
        curr_lr = optimizer.param_groups[0]['lr']
        data_time.update(time.time() - end)
        views = batch[0]  # list of tensors
        bs = len(batch[1])  # num shapes in GPU batch
        if args.simclr:
            # self-sup contrastive learning
            # TOCHECK: don't want samples on diff. GPU workers to have the same label
            targets = torch.tensor(range(bs)) + rank * bs
        else:
            # supervised contrastive learning
            targets = batch[1]

        targets = targets.cuda(non_blocking=True)
        targets = targets.repeat(len(views))  # bs -> bs*num_views

        assert isinstance(views, list), "expected list of object views"
        points = torch.cat(views, dim=0).cuda(non_blocking=True)  # [bs*num_views,num_points,3]

        """ Single optimization step """
        with autocast(enabled=args.use_amp):
            z = model(points)  # [bs * num_views, emb_dim]
            loss = simclr_loss_func(z, indexes=targets, temperature=args.temp)

        scaler.scale(loss).backward()

        if args.grad_norm_clip > 0:
            # grad clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm_clip, norm_type=2)
            raise NotImplementedError("grad_norm_clip")

        scaler.step(optimizer)
        scaler.update()

        # logging
        losses.update(loss.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            log_str = f"it: [{i + 1}/{len(loader)}-{epoch}/{args.epochs}], rank: [{rank + 1}/{ws}], " \
                      f"Loss: {losses.val:.4f}, Loss avg: {losses.avg:.4f}, LR: {curr_lr:.6f}, " \
                      f"BT: {batch_time.val:.2f}, DT: {data_time.val:.2f}"
            if rank == 0:
                io_logger.cprint(log_str)
                wandb.log({"train/it_loss": loss.item(), "train/it_lr": curr_lr, 'train/it': args.glob_it})
            else:
                print(log_str)

    time2 = time.time()
    epoch_loss = losses.avg
    log_str = f"Train Epoch: {epoch}, rank: [{rank + 1}/{ws}], " \
              f"Loss: {epoch_loss:.4f}, " \
              f"BT: {batch_time.avg:.2f}, DT: {data_time.avg:.2f},  " \
              f"time: {(time2 - time1):.2f}"

    if rank == 0:
        io_logger.cprint(log_str)
        wandb.log({"train/epoch_loss": epoch_loss, "train/epoch": epoch})  # loss referred to a single GPU worker
    else:
        print(log_str)

    return epoch_loss

def trainer(opt, config):
    dist.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opt.local_rank, torch.device(opt.local_rank)
    rank, world_size = get_rank(), get_ws()
    assert torch.cuda.is_available(), "no cuda device is available"
    torch.cuda.set_device(device_id)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(opt.seed)

    print("*" * 30)
    print(f"{rank}/{world_size} process initialized.\n")
    print(f"{rank}/{world_size} arguments: {opt}. \n")
    print("*" * 30)

    assert opt.config is not None and osp.exists(opt.config)

    mode = "SUPCON"
    if opt.simclr:
        mode = "SIMCLR"

    # setup loggers
    if rank == 0:
        safe_make_dirs([opt.models_dir, opt.tb_dir, opt.backup_dir])
        project_dir = os.getcwd()  # osp.dirname(os.path.abspath(__file__))
        os.system('cp {} {}/'.format(osp.abspath(__file__), opt.backup_dir))
        os.system('cp -r {} {}/config.yaml'.format(opt.config, opt.backup_dir))
        os.system('cp -r {} {}/'.format(osp.join(project_dir, "models"), opt.backup_dir))
        os.system('cp -r {} {}/'.format(osp.join(project_dir, "datasets"), opt.backup_dir))
        logger = IOStream(path=osp.join(opt.log_dir, f'log_{int(time.time())}.txt'))
        logger.cprint(f"Arguments: {opt}")
        logger.cprint(f"Config: {config}")
        logger.cprint(f"World size: {world_size}")
        logger.cprint(f"MODE: {mode}\n")
        wandb.login()
        if opt.wandb_name is None:
            opt.wandb_name = opt.exp_name
        wandb.init(project=opt.wandb_proj, group=opt.wandb_group, name=opt.wandb_name,
                   config={'arguments': vars(opt), 'config': config})
    else:
        logger = None

    # training dataloader
    train_loader = get_train_loader_dpp(opt)

    # dataloaders for evaluation
    src_name, target_names, noAugm_train_loader, src_loader, tar1_loader, tar2_loader = get_test_loaders_ddp(opt)

    n_classes = train_loader.dataset.num_classes
    # build model
    model = SimCLR(DotConfig(config['model'])).cuda()
    model.apply(weights_init_normal)
    print(f"model: \n{model}")

    if opt.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f"Model params count: {count_parameters(model)/(1000000):.4f} M")
    model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    if rank == 0:
        wandb.watch(model, criterion=None, log="gradients", log_freq=100, idx=None, log_graph=True)

    # optimizer and scheduler
    optimizer, scheduler = get_opti_sched(model.named_parameters(), config)
    scaler = GradScaler(enabled=opt.use_amp)
    start_epoch = 1
    glob_it = 0
    if opt.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % device_id}  # configure map_location properly
        ckt = torch.load(opt.resume, map_location=map_location)
        print(f"rank {rank} load weights: ", model.load_state_dict(ckt['model'], strict=True))
        optimizer.load_state_dict(ckt['optimizer'])
        scheduler.load_state_dict(ckt['scheduler'])
        if opt.use_amp:
            assert 'scaler' in ckt.keys(), "No scaler key in ckt"
            assert ckt['scaler'] is not None, "None scaler object in ckt"
            scaler.load_state_dict(ckt['scaler'])
        if rank == 0:
            logger.cprint("Restart training from checkpoint %s" % opt.resume)
        start_epoch += int(ckt['epoch'])
        glob_it += (int(ckt['epoch']) * len(train_loader))
        del ckt

    # TRAINER
    opt.glob_it = glob_it  # will be update by the train_epoch fun.
    time1 = time.time()
    for epoch in range(start_epoch, opt.epochs + 1):
        if isinstance(train_loader, DataLoader) and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        else:
            assert False

        train_epoch_supcon(
            epoch=epoch,
            args=opt,
            loader=train_loader,
            model=model,
            scaler=scaler,
            optimizer=optimizer,
            io_logger=logger
        )

        # step lr
        scheduler.step(epoch)

        # runtime OOD evaluation
        if epoch % opt.eval_step == 0 or epoch >= (int(opt.epochs) - 10):
            noAugm_train_loader.sampler.set_epoch(epoch)
            src_loader.sampler.set_epoch(epoch)
            tar1_loader.sampler.set_epoch(epoch)
            tar2_loader.sampler.set_epoch(epoch)
            start_eval = time.time()
            train_centroids = compute_centroids(model, noAugm_train_loader)
            id_preds, id_labels = compute_clf_centroids(model, train_centroids, src_loader)

            # each list should be composed as [SRC, TAR1, TAR2]
            scores_list = compute_sim_centroids(model, train_centroids, [src_loader, tar1_loader, tar2_loader])
            preds_list = [id_preds, None, None]
            labels_list = [id_labels, None, None]
            test_acc, test_bal_acc, res_tar1, res_tar2, res_tar3 = eval_ood_sncore(
                scores_list, preds_list, labels_list, src_label=1, silent=True)
            auroc1, fpr1 = res_tar1['auroc'], res_tar1['fpr_at_95_tpr']
            auroc2, fpr2 = res_tar2['auroc'], res_tar2['fpr_at_95_tpr']
            auroc3, fpr3 = res_tar3['auroc'], res_tar3['fpr_at_95_tpr']
            eval_time = time.time() - start_eval
            if rank == 0:
                wandb.log({
                    "test/acc": test_acc, "test/bal_acc": test_bal_acc,
                    "test/auroc1": auroc1, "test/fpr1": fpr1,
                    "test/auroc2": auroc2, "test/fpr2": fpr2,
                    "test/auroc3": auroc3, "test/fpr3": fpr3,
                    "test/epoch": epoch
                })
                logger.cprint(
                    f"Test [{epoch}/{opt.epochs}]\tAcc: {test_acc:.4f}, Bal Acc: {test_bal_acc:.4f},\n"
                    f"\t{src_name}->{target_names[0]} AUROC: {auroc1:.4f}, FPR95: {fpr1:.4f},\n"
                    f"\t{src_name}->{target_names[1]} AUROC: {auroc2:.4f}, FPR95: {fpr2:.4f},\n"
                    f"\t{src_name}->{target_names[0]}+{target_names[1]} AUROC: {auroc3:.4f}, FPR95: {fpr3:.4f},\n"
                    f"eval time: {eval_time:.2f}", color='b')

        # save checkpoint
        if rank == 0:
            if epoch % opt.save_step == 0:
                ckt_path = osp.join(opt.models_dir, "model_last.pth")
                save_checkpoint(opt, ckt_path, model, optimizer, scheduler, scaler, config, epoch)
            if epoch % 100 == 0 or epoch >= (int(opt.epochs) - 10):
                ckt_path = osp.join(opt.models_dir, f"model_{epoch}.pth")
                save_checkpoint(opt, ckt_path, model, optimizer, scheduler, scaler, config, epoch)

    time2 = time.time()
    if rank == 0:
        logger.cprint(
            f"Training finished - elapsed time: {time2 - time1}")


def evaluator(opt, config):
    """
    single GPU evaluation routine
    """
    assert torch.cuda.is_available(), "no cuda device is available"
    print(f"Arguments: {opt}")
    set_random_seed(opt.seed)

    # batch size fixed to 32
    train_loader, (src_loader, tar1_loader, tar2_loader), (src_name, tar1_name, tar2_name)\
        = get_sncore_id_ood_loaders(opt)

    n_classes = train_loader.dataset.num_classes
    # build model
    model = SimCLR(DotConfig(config['model']))

    assert opt.ckpt_path is not None and osp.exists(opt.ckpt_path)
    print(f"Resuming model from ckt: {opt.ckpt_path}")
    ckt = torch.load(opt.ckpt_path, map_location='cpu')
    print(f"Ckt from epoch: {ckt['epoch']}")
    resume_dict = sanitize_model_dict(ckt['model'])  # remove 'module' prefix
    print("load weights: ", model.load_state_dict(resume_dict, strict=False))
    del ckt

    print(f"Model params count: {count_parameters(model)/(1000000):.4f} M")

    # move to GPU + eval mode
    model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()

    print("\n")
    print("#" * 80)
    print("Evaluation - Category Centroids (computed on training set)\n")

    train_centroids = compute_centroids(model, train_loader)

    #############################
    # Category/Model Centroids eval
    id_preds, id_labels = compute_clf_centroids(model, train_centroids, src_loader)  # preds, gt for in-dist samples
    # each list should be composed as [SRC, TAR1, TAR2]
    scores_list = compute_sim_centroids(model, train_centroids, [src_loader, tar1_loader, tar2_loader])
    preds_list = [id_preds, None, None]
    labels_list = [id_labels, None, None]
    eval_ood_sncore(scores_list, preds_list, labels_list, src_label=1)
    print("#" * 80)
    print("\n")

    #############################
    # Nearest training sample
    print("#" * 80)
    print("Evaluation - Nearest Training Sample\n")
    print(f"Src: {src_name}, Tar 1: {tar1_name}, Tar 2: {tar2_name}, Tar 3: {tar1_name}+{tar2_name}")
    eval_ood_sncore_csi(model, train_loader, src_loader, tar1_loader, tar2_loader, use_norm=False)
    print("#" * 80)
    print("\n")

    #############################
    # CSI (Nearest training sample * norm)
    print("#" * 80)
    print("Evaluation - CSI\n")
    print(f"Src: {src_name}, Tar 1: {tar1_name}, Tar 2: {tar2_name}, Tar 3: {tar1_name}+{tar2_name}")
    eval_ood_sncore_csi(model, train_loader, src_loader, tar1_loader, tar2_loader, use_norm=True)
    print("#" * 80)
    print("\n")

    #############################
    # Euclidean dist
    print("#" * 80)
    print("Evaluation - Euclidean dist\n")
    print(f"Src: {src_name}, Tar 1: {tar1_name}, Tar 2: {tar2_name}, Tar 3: {tar1_name}+{tar2_name}")
    from knn_cuda import KNN
    knn = KNN(k=1, transpose_mode=True)

    train_feats, _ = get_simclr_proj(model, train_loader)
    src_feats, _ = get_simclr_proj(model, src_loader)
    tar1_feats, _ = get_simclr_proj(model, tar1_loader)
    tar2_feats, _ = get_simclr_proj(model, tar2_loader)

    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(train_feats.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = 1 / src_dist

    # OOD tar1
    tar1_dist, _ = knn(train_feats.unsqueeze(0), tar1_feats.unsqueeze(0))
    tar1_dist = tar1_dist.squeeze().cpu()
    tar1_scores = 1 / tar1_dist

    # OOD tar2
    tar2_dist, _ = knn(train_feats.unsqueeze(0), tar2_feats.unsqueeze(0))
    tar2_dist = tar2_dist.squeeze().cpu()
    tar2_scores = 1 / tar2_dist

    eval_ood_sncore(
        scores_list=[src_scores, tar1_scores, tar2_scores],
        preds_list=[None, None, None],
        labels_list=[None, None, None],
        src_label=1  # confidence should be higher for ID samples
    )


    print("#" * 80)
    print("\n")


def main():
    args = get_args()
    args.data_root = os.path.expanduser(args.data_root)

    config = load_yaml(args.config)
    mode = str(args.script_mode)

    if mode == 'train':
        # launch trainer
        print("training..")
        assert args.checkpoints_dir is not None and len(args.checkpoints_dir)
        assert args.exp_name is not None and len(args.exp_name)
        args.log_dir = osp.join(args.checkpoints_dir, args.exp_name)
        args.tb_dir = osp.join(args.checkpoints_dir, args.exp_name, "tb-logs")
        args.models_dir = osp.join(args.checkpoints_dir, args.exp_name, "models")
        args.backup_dir = osp.join(args.checkpoints_dir, args.exp_name, "backup-code")
        trainer(args, config)
    elif mode.startswith('eval'):
        print("out-of-distribution evaluation..")
        evaluator(args, config)
    else:
        raise ValueError(f"Unknown script mode: {args.script_mode}")


if __name__ == '__main__':
    main()
