import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.multiprocessing
import time
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm as sbn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import *
from utils.dist import *
from utils.data_utils import H5_Dataset
from datasets.modelnet import *
from datasets.scanobject import *
# noinspection PyUnresolvedReferences
from models.density import build_nf_head, build_cls_head, Encoder, get_nll_loss, get_ll
from utils.ood_utils import get_confidence, eval_ood_sncore, get_ood_metrics
import wandb
from base_args import add_base_args
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from models.common import convert_model_state
from utils.utils import cal_ce_loss
from classifiers.trainer_ddp_cla_r2r import get_loaders_train, get_loaders_test


def get_args():
    parser = argparse.ArgumentParser("OOD on point clouds via contrastive learning")
    parser = add_base_args(parser)
    # experiment specific args
    parser.add_argument("--augm_set",
                        type=str, default="rw", help="data augmentation choice", choices=["st", "rw"])
    parser.add_argument("--grad_norm_clip",
                        default=-1, type=float, help="gradient clipping")
    parser.add_argument("--num_points",
                        default=2048, type=int, help="number of points sampled for each object view")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="real2real_NF")
    parser.add_argument("--wandb_proj", type=str, default="benchmark-3d-ood-cla")
    parser.add_argument("--loss", type=str, default="CE",
                        choices=["CE", "CE_ls"],
                        help="Which loss to use for training. CE is default")
    parser.add_argument("--save_feats", type=str, default=None,
                        help="Path where to save feats of penultimate layer")
    # encoder + normalizing flow
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate normalizing flow head')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay normalizing flow head')
    parser.add_argument("--flow_step", default=1, type=int,
                        help="Optimization step normalizing flow head")
    # Adopt Corrupted data
    # this flag should be set also during evaluation if testing Synth->Real Corr/LIDAR Augmented models
    parser.add_argument("--corruption",
                        type=str, default=None, help="type of corrupted data (lidar,occlusion,all) - default is None")

    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)
    args.tar1 = "none"
    args.tar2 = "none"
    if args.script_mode == 'eval':
        args.batch_size = 1
    return args



@torch.no_grad()
def compute_loglikelihood(encoder, nf_head, dataloader):
    """ DDP impl """
    all_lls = []
    all_labels = []
    encoder.eval()
    nf_head.eval()
    for _, batch in enumerate(dataloader, 0):
        points, targets = batch[0], batch[1]
        points = points.cuda(non_blocking=True)

        # get lls
        feat = encoder(points)
        z = nf_head(feat)
        lls = get_ll(z, nf_head.jacobian(run_forward=False))

        if is_dist() and get_ws() > 1:
            lls = gather(lls, dim=0)
            targets = gather(targets, dim=0)

        all_lls.append(lls)
        all_labels.append(targets)
    all_lls = torch.cat(all_lls, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_lls, all_labels


@torch.no_grad()
def compute_clf(encoder, cls_head, dataloader, softmax=True):
    """ DDP impl """
    all_conf = []
    all_pred = []
    all_labels = []
    encoder.eval()
    cls_head.eval()
    for _, batch in enumerate(dataloader, 0):
        points, targets = batch[0], batch[1]
        points = points.cuda(non_blocking=True)

        feat = encoder(points)
        logits = cls_head(feat)

        if is_dist() and get_ws() > 1:
            logits = gather(logits, dim=0)
            targets = gather(targets, dim=0)

        probs = F.softmax(logits, 1) if softmax else logits
        conf, pred = probs.data.max(1)
        all_conf.append(conf)
        all_pred.append(pred)
        all_labels.append(targets)
    all_conf = torch.cat(all_conf, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_conf, all_pred, all_labels


def train_epoch_hybrid(epoch, args, loader, encoder, cls_head, nf_head, opti, nf_opti, logger):
    """
    Training the encoder and cls_head: optimizer is 'opti'
    Training the nf_head: optimizer is 'nf_opti'
    nf_head is separately trained (is not influencing neither encoder neither the cls_head)
    """
    train_preds = []
    train_targets = []
    rank = get_rank()
    ws = get_ws()
    time1 = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if args.use_amp:
        raise NotImplementedError("amp not implemented")

    if args.grad_norm_clip > 0:
        raise NotImplementedError("grad_norm_clip not implemented")

    # train all
    encoder.train()
    cls_head.train()
    nf_head.train()

    for it, batch in enumerate(loader, 1):
        args.glob_it += 1
        clf_lr = opti.param_groups[0]['lr']
        nf_lr = nf_opti.param_groups[0]['lr']
        data_time.update(time.time() - end)
        points = batch[0].cuda(non_blocking=True)  # [bs,num_points,3]
        targets = batch[1].cuda(non_blocking=True)  # [bs,]
        bs = len(targets)  # num shapes in GPU batch
        nll_loss = torch.Tensor([0.0])

        # train cross-entropy (encoder+clf head)
        feat = encoder(points)
        cls_logits = cls_head(feat)
        ce_loss = cal_ce_loss(pred=cls_logits, target=targets, smoothing=(args.loss == "CE_ls"))
        opti.zero_grad()
        ce_loss.backward()
        opti.step()

        # cross-entropy log
        preds = to_numpy(cls_logits.max(dim=1)[1])
        targets = to_numpy(targets)
        train_preds.append(preds)
        train_targets.append(targets)

        # train normalizing flow head
        if it % args.flow_step:
            opti.zero_grad()  # should be useless
            nf_opti.zero_grad()
            with torch.no_grad():
                feat = encoder(points)
            z = nf_head(feat)
            nll_loss = get_nll_loss(z, nf_head.module.jacobian(run_forward=False))
            nll_loss.backward()
            nf_opti.step()

        # logging
        batch_time.update(time.time() - end)
        end = time.time()
        if it % args.flow_step:
            log_str = f"it: [{it}/{len(loader)}-{epoch}/{args.epochs}], rank: [{rank + 1}/{ws}], " \
                      f"ce_loss: {ce_loss.item():.4f}, nll_loss: {nll_loss.item():.4f}, " \
                      f"clf_lr: {clf_lr:.6f}, nf_lr: {nf_lr:.6f}, BT: {batch_time.val:.2f}, DT: {data_time.val:.2f}"
            if rank == 0:
                logger.cprint(log_str)
                wandb.log({"train/ce_loss": ce_loss.item(), "train/nll_loss": nll_loss.item(), "train/cls_lr": clf_lr,
                           "train/nf_lr": nf_lr, "train/it": args.glob_it})
            else:
                print(log_str)

    time2 = time.time()
    train_targets = np.concatenate(train_targets, 0)
    train_preds = np.concatenate(train_preds, 0)
    epoch_acc = accuracy_score(train_targets, train_preds)
    epoch_bal_acc = balanced_accuracy_score(train_targets, train_preds)
    log_str = f"Train [{epoch}/{args.epochs}]\t" \
              f"rank: [{rank + 1}/{ws}], " \
              f"Acc: {epoch_acc:.4f}, " \
              f"Bal Acc: {epoch_bal_acc:.4f}, " \
              f"BT: {batch_time.avg:.2f}, " \
              f"DT: {data_time.avg:.2f},  " \
              f"epoch time: {(time2 - time1):.2f}\n"
    res_epoch = {"train/ep_acc": epoch_acc, "train/ep_bal_acc": epoch_bal_acc, "train/epoch": epoch}
    if rank == 0:
        logger.cprint(log_str)
        wandb.log(res_epoch)  # single GPU worker
    else:
        print(log_str)

    return res_epoch


def train(opt, config):
    dist.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opt.local_rank, torch.device(opt.local_rank)
    rank, world_size = get_rank(), get_ws()
    assert torch.cuda.is_available(), "no cuda device is available"
    torch.cuda.set_device(device_id)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(opt.seed)
    print("*" * 30)
    print(f"{rank}/{world_size} process initialized.\n")
    print(f"{rank}/{world_size} arguments: {opt}. \n")
    print("*" * 30)

    # setup loggers
    if rank == 0:
        safe_make_dirs([opt.models_dir, opt.tb_dir, opt.backup_dir])
        project_dir = os.getcwd()
        os.system('cp {} {}/'.format(osp.abspath(__file__), opt.backup_dir))
        os.system('cp -r {} {}/'.format(opt.config, opt.backup_dir))
        os.system('cp -r {} {}/'.format(osp.join(project_dir, "models"), opt.backup_dir))
        os.system('cp -r {} {}/'.format(osp.join(project_dir, "datasets"), opt.backup_dir))
        logger = IOStream(path=osp.join(opt.log_dir, f'log_{int(time.time())}.txt'))
        logger.cprint(f"Arguments: {opt}")
        logger.cprint(f"Config: {config}")
        logger.cprint(f"World size: {world_size}\n")
        wandb.login()
        if opt.wandb_name is None:
            opt.wandb_name = opt.exp_name
        wandb.init(project=opt.wandb_proj, group=opt.wandb_group, name=opt.wandb_name,
                   config={'arguments': vars(opt), 'config': config})
    else:
        logger = None

    # training dataloader
    train_loader, _, test_loader = get_loaders_train(opt)

    # build model
    train_synset = eval(opt.src)
    if rank == 0:
        logger.cprint(f"{opt.src} train synset: {train_synset}")
    n_classes = len(set(train_synset.values()))

    if rank == 0:
        logger.cprint(f"Source: {opt.src}\n"
                      f"Num training classes: {n_classes}")

    # build model
    encoder = Encoder(args=DotConfig(config['model'])).cuda()
    nf_head = build_nf_head(input_dim=512).cuda()
    cls_head = build_cls_head(input_dim=512, num_classes=n_classes, args=DotConfig(config['model'])).cuda()
    encoder.apply(weights_init_normal)
    nf_head.apply(weights_init_normal)
    cls_head.apply(weights_init_normal)

    tot_params = count_parameters(encoder) + count_parameters(nf_head) + count_parameters(cls_head)
    print(f"Model params count: {tot_params / 1000000:.4f} M")

    if opt.use_sync_bn:
        encoder = sbn.convert_sync_batchnorm(encoder)
        nf_head = sbn.convert_sync_batchnorm(nf_head)
        cls_head = sbn.convert_sync_batchnorm(cls_head)

    # parallelize
    encoder = DDP(encoder, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    cls_head = DDP(cls_head, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    nf_head = DDP(nf_head, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)

    if rank == 0:
        wandb.watch(encoder, log='gradients')
        wandb.watch(nf_head, log='gradients')
        wandb.watch(cls_head, log='gradients')

    ########################
    # build optimizers
    # Encoder and Classification Head
    clf_params = list(encoder.named_parameters()) + list(cls_head.named_parameters())
    optimizer, scheduler = get_opti_sched(clf_params, config)
    # Normalizing Flow Head
    nf_optimizer = optim.Adam(nf_head.parameters(), lr=opt.lr, weight_decay=opt.wd, betas=(0.8, 0.8), eps=1e-4)
    ########################

    start_epoch = 1
    glob_it = 0
    if opt.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # configure map_location properly
        ckt = torch.load(opt.resume, map_location=map_location)
        encoder.load_state_dict(ckt['encoder'], strict=True)
        nf_head.load_state_dict(ckt['nf_head'], strict=True)
        cls_head.load_state_dict(ckt['cls_head'], strict=True)
        scheduler.load_state_dict(ckt['scheduler'])
        optimizer.load_state_dict(ckt['optimizer'])
        nf_optimizer.load_state_dict(ckt['nf_optimizer'])
        if rank == 0:
            logger.cprint("Restart training from checkpoint %s" % opt.resume)
        start_epoch += int(ckt['epoch'])
        glob_it += (int(ckt['epoch']) * len(train_loader))
        del ckt

    # TRAINER
    opt.glob_it = glob_it  # will be update by the train_epoch fun.
    best_epoch, best_acc = -1, -1
    time1 = time.time()
    for epoch in range(start_epoch, opt.epochs + 1):
        is_best = False
        if isinstance(train_loader, DataLoader) and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_epoch_hybrid(epoch=epoch, args=opt, loader=train_loader, encoder=encoder, cls_head=cls_head,
                           nf_head=nf_head, opti=optimizer, nf_opti=nf_optimizer, logger=logger)

        # step lr
        scheduler.step(epoch)

        # save checkpoint
        if rank == 0:
            # save last
            ckt_path = osp.join(opt.models_dir, "model_last.pth")
            torch.save({
                'encoder': encoder.state_dict(), 'nf_head': nf_head.state_dict(), 'cls_head': cls_head.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'nf_optimizer': nf_optimizer.state_dict(), 'args': opt, 'config': config, 'epoch': epoch}, ckt_path)
            if is_best:
                os.system('cp -r {} {}'.format(ckt_path, osp.join(opt.models_dir, f"model_best.pth")))
            if epoch % opt.save_step == 0:
                os.system('cp -r {} {}'.format(ckt_path, osp.join(opt.models_dir, f"model_ep{epoch}.pth")))

    time2 = time.time()
    if rank == 0:
        logger.cprint(
            f"Training finished - best test acc: {best_acc:.4f} at epoch: {best_epoch}, time: {time2 - time1}")


def eval_ood(opt, config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"Arguments: {opt}")
    set_random_seed(opt.seed)

    # src data loaders
    train_loader, _, test_loader, target_loader = get_loaders_test(opt)
    id_loader = test_loader  # known classes test samples
    ood_loader = target_loader  # unknown classes test samples

    classes_dict = eval(opt.src)
    n_classes = len(set(classes_dict.values()))
    encoder = Encoder(args=DotConfig(config['model']))
    nf_head = build_nf_head(input_dim=512)
    cls_head = build_cls_head(input_dim=512, num_classes=n_classes, args=DotConfig(config['model']))
    ckt = torch.load(opt.ckpt_path, map_location='cpu')
    enco_weights = sanitize_model_dict(ckt['encoder'])
    cls_weights = sanitize_model_dict(ckt['cls_head'])
    nf_weights = sanitize_model_dict(ckt['nf_head'])
    print("load encoder: ", encoder.load_state_dict(enco_weights, strict=True))
    print("load cls head: ", cls_head.load_state_dict(cls_weights, strict=True))
    print("load nf head: ", nf_head.load_state_dict(nf_weights, strict=True))
    del enco_weights, cls_weights, nf_weights, ckt

    encoder = encoder.cuda()
    cls_head = cls_head.cuda()
    nf_head = nf_head.cuda()

    # source
    src_ll, src_labels = compute_loglikelihood(encoder=encoder, nf_head=nf_head, dataloader=id_loader)
    src_conf, src_pred, src_labels = compute_clf(encoder=encoder, cls_head=cls_head, dataloader=id_loader)

    # target
    tar_ll, tar_labels = compute_loglikelihood(encoder=encoder, nf_head=nf_head, dataloader=ood_loader)
    tar_conf, tar_pred, tar_labels = compute_clf(encoder=encoder, cls_head=cls_head, dataloader=ood_loader)

    # compute test accuracy
    import sklearn.metrics as skm
    src_labels = to_numpy(src_labels)
    src_pred = to_numpy(src_pred)
    src_acc = skm.accuracy_score(src_labels, src_pred)
    src_bal_acc = skm.balanced_accuracy_score(src_labels, src_pred)
    print(f"Src Test - Acc: {src_acc}, Bal Acc: {src_bal_acc}\n")

    # MSP
    print("\n" + "#" * 80)
    print("Computing OOD metrics with MSP normality score...")
    res = get_ood_metrics(src_conf, tar_conf, src_label=1)
    print(res)
    print("#" * 80)

    # LogLikelihood OOD
    print("#" * 80 + "\n")
    print("Computing OOD metrics with loglikelihood normality score...")
    res = get_ood_metrics(src_ll, tar_ll, src_label=1)
    print(res)
    print("#" * 80)
    return


def main():
    args = get_args()
    assert args.config is not None and osp.exists(args.config)
    config = load_yaml(args.config)
    if args.script_mode == 'train':
        # launch trainer
        print("training...")
        assert args.checkpoints_dir is not None and len(args.checkpoints_dir)
        assert args.exp_name is not None and len(args.exp_name)
        args.log_dir = osp.join(args.checkpoints_dir, args.exp_name)
        args.tb_dir = osp.join(args.checkpoints_dir, args.exp_name, "tb-logs")
        args.models_dir = osp.join(args.checkpoints_dir, args.exp_name, "models")
        args.backup_dir = osp.join(args.checkpoints_dir, args.exp_name, "backup-code")
        train(args, config)
    elif args.script_mode == 'eval':
        print("out-of-distribution evaluation...")
        assert args.ckpt_path is not None and len(args.ckpt_path)
        eval_ood(args, config)
    else:
        raise ValueError(f"Unknown script mode: {args.script_mode}")


if __name__ == '__main__':
    main()
