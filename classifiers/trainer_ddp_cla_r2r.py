# real -> real experiments
import sys
import os
sys.path.append(os.getcwd())
import time
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import *
from utils.dist import *
# noinspection PyUnresolvedReferences
from utils.data_utils import H5_Dataset
from datasets.modelnet import *
from datasets.scanobject import *
from models.classifiers import Classifier
from utils.ood_utils import get_confidence, eval_ood_sncore, iterate_data_odin, \
    iterate_data_energy, iterate_data_gradnorm, iterate_data_react, estimate_react_thres, print_ood_output, \
    get_penultimate_feats, get_network_output
import wandb
from base_args import add_base_args
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from models.common import convert_model_state, logits_entropy_loss
from models.ARPL_utils import Generator, Discriminator
from classifiers.common import train_epoch_cla, train_epoch_rsmix_exposure, train_epoch_cs
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from utils.ood_utils import get_ood_metrics


# REAL -> REAL experiments


def get_args():
    parser = argparse.ArgumentParser("OOD on point clouds via contrastive learning")
    parser = add_base_args(parser)

    # experiment specific arguments
    parser.add_argument("--augm_set",
                        type=str, default="rw", help="data augmentation choice", choices=["st", "rw"])
    parser.add_argument("--grad_norm_clip",
                        default=-1, type=float, help="gradient clipping")
    parser.add_argument("--num_points",
                        default=2048, type=int, help="number of points sampled for each object view")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="real2real")
    parser.add_argument("--wandb_proj", type=str, default="benchmark-3d-ood-cla")
    parser.add_argument("--loss", type=str, default="CE",
                        choices=["CE", "CE_ls", "cosface", "arcface", "subcenter_arcface", "ARPL", "cosine"],
                        help="Which loss to use for training. CE is default")
    #
    parser.add_argument("--cs", action='store_true', help="Enable confusing samples for ARPL")
    parser.add_argument("--cs_gan_lr", type=float, default=0.0002, help="Confusing samples GAN lr")
    parser.add_argument("--cs_beta", type=float, default=0.1, help="Beta loss weight for CS")
    parser.add_argument("--save_feats", type=str, default=None, help="Path where to save feats of penultimate layer")
    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)
    args.tar1 = "none"
    args.tar2 = "none"

    if args.script_mode == 'eval':
        args.batch_size = 1

    return args


### data mgmt ###

# for training routine
def get_loaders_train(opt):
    ws, rank = get_ws(), get_rank()
    print(f"Train - num_points: {opt.num_points}")

    sonn_args = {
        'data_root': opt.data_root,
        'sonn_split': opt.sonn_split,
        'h5_file': opt.sonn_h5_name,
    }

    if opt.augm_set == 'rw':
        # transformation used for Synthetic->Real-World
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),  # sampling as a data augmentation
            AugmScale(),
            AugmRotate(axis=[0.0, 1.0, 0.0]),
            AugmRotatePerturbation(),
            AugmTranslate(),
            AugmJitter()]
    else:
        raise ValueError(f"Real->Real - Wrong augmentation set: {opt.augm_set}")
    print("Train - transforms: ", set_transforms)

    #############
    # src train #
    #############
    whole_train_data = ScanObject(  # sampling performed as a data augm. during training
        **sonn_args,
        num_points=2048,  # means 'take all points': use transforms to eventually perform sampling!
        split='train',
        class_choice=opt.src,
        transforms=transforms.Compose(set_transforms))

    # split whole train into train and val (deterministic)
    num_val = int(len(whole_train_data) * 10 / 100)
    train_idx, val_idx = train_test_split(list(range(len(whole_train_data))), test_size=num_val, shuffle=True, random_state=42)
    train_data = Subset(whole_train_data, train_idx)
    val_data = Subset(whole_train_data, val_idx)  # val_data is augmented as train_data during training

    # src test
    test_data = ScanObject(
        **sonn_args,
        num_points=opt.num_points,  # same as training
        split='test',
        class_choice=opt.src,
        transforms=None)

    print(f"Train - src: {opt.src} - train_data: {len(train_data)}, val_data: {len(val_data)}, test_data: {len(test_data)}")
    train_sampler = DistributedSampler(train_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    test_sampler = DistributedSampler(test_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers,
                              sampler=train_sampler, worker_init_fn=init_np_seed, shuffle=train_sampler is None)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers,
                            sampler=val_sampler, worker_init_fn=init_np_seed, shuffle=val_sampler is None)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers,
                             sampler=test_sampler, worker_init_fn=init_np_seed, shuffle=test_sampler is None)

    """
    # tar data
    if opt.src == "SR12":
        target = "sonn_ood_common"
    elif opt.src == "SR13":
        target = "sonn_2_mdSet2"
    elif opt.src == "SR23":
        target = "sonn_2_mdSet1"
    else:
        raise ValueError(f"Unknown source: {opt.src}")

    # TODO: using all data samples (train+test) for unknown target
    # TODO: 2048 points hardcoded
    target_data = ScanObject(
        data_root=opt.data_root,
        sonn_split='main_split',
        h5_file='objectdataset.h5',
        num_points=2048,
        split='all',
        class_choice=target,
        transforms=None)
    print(f"Train - target: {target} - target_data: {len(target_data)}")

    train_sampler = DistributedSampler(train_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    test_sampler = DistributedSampler(test_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    target_sampler = DistributedSampler(target_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
                              sampler=train_sampler, worker_init_fn=init_np_seed, shuffle=train_sampler is None)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
                            sampler=val_sampler, worker_init_fn=init_np_seed, shuffle=val_sampler is None)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
                             sampler=test_sampler, worker_init_fn=init_np_seed, shuffle=test_sampler is None)
    tar_loader = DataLoader(target_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
                            sampler=target_sampler, worker_init_fn=init_np_seed, shuffle=target_sampler is None)

    return train_loader, val_loader, test_loader, tar_loader
    """

    return train_loader, val_loader, test_loader


# for evaluation routine
def get_loaders_test(opt):
    assert str(opt.script_mode).startswith('eval')
    sonn_args_test = {
        'data_root': opt.data_root,
        'sonn_split': opt.sonn_split,
        'h5_file': opt.sonn_h5_name,
        'transforms': None,
        'num_points': opt.num_points,
    }

    loader_args_test = {
        'batch_size': opt.batch_size,
        'drop_last': False,
        'shuffle': False,
        'num_workers': opt.num_workers,
        'worker_init_fn': init_np_seed,
    }

    print(f"Test - num_points: {opt.num_points}")

    #############
    # SRC Train #
    #############
    whole_train_data = ScanObject(split='train', class_choice=opt.src, **sonn_args_test)

    # split train val is deterministic
    num_val = int(len(whole_train_data) * 10 / 100)
    train_idx, val_idx = train_test_split(
        list(range(len(whole_train_data))), test_size=num_val, shuffle=True, random_state=42)
    train_data = Subset(whole_train_data, train_idx)
    val_data = Subset(whole_train_data, val_idx)


    ############
    # SRC Test #
    ############
    test_data = ScanObject(split='test', class_choice=opt.src, **sonn_args_test)
    print(f"Test - src: {opt.src} - train_data: {len(train_data)}, val_data: {len(val_data)}, test_data: {len(test_data)}")


    ############
    # TAR Test #
    ############
    # target choice depending on opt.src
    if opt.src == "SR12":
        target_name = "sonn_ood_common"
    elif opt.src == "SR13":
        target_name = "sonn_2_mdSet2"
    elif opt.src == "SR23":
        target_name = "sonn_2_mdSet1"
    else:
        raise ValueError(f"Unknown source: {opt.src}")
    target_data = ScanObject(split='all', class_choice=target_name, **sonn_args_test)
    print(f"Test - target: {target_name} - target_data: {len(target_data)}")

    train_loader = DataLoader(train_data, **loader_args_test)
    val_loader = DataLoader(val_data, **loader_args_test)
    test_loader = DataLoader(test_data, **loader_args_test)
    target_loader = DataLoader(target_data, **loader_args_test)
    return train_loader, val_loader, test_loader, target_loader


##############################


def train(opt, config):
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

    assert str(opt.src).startswith("SR"), f"Unknown src choice: {opt.src}"
    train_loader, _, test_loader = get_loaders_train(opt)
    train_synset = eval(opt.src)
    n_classes = len(set(train_synset.values()))

    if rank == 0:
        logger.cprint(f"Source: {opt.src} \nNum classes: {n_classes} \nsynset: {train_synset}")

    # BUILD MODEL
    model = Classifier(args=DotConfig(config['model']), num_classes=n_classes, loss=opt.loss, cs=opt.cs)
    enco_name = str(config['model']['ENCO_NAME']).lower()
    if enco_name == "gdanet":
        model.apply(weight_init_GDA)
    else:
        model.apply(weights_init_normal)

    model = model.cuda()
    if opt.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if rank == 0:
        logger.cprint(f"Model: \n{model}\n")
        logger.cprint(f"param count: \n{count_parameters(model) / 1000000 :.4f} M")
        logger.cprint(f"Loss: {opt.loss}\n")

    model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    if rank == 0:
        wandb.watch(model, log="gradients")

    # optimizer and scheduler
    optimizer, scheduler = get_opti_sched(model.named_parameters(), config)
    scaler = GradScaler(enabled=opt.use_amp)
    netG, netD = None, None
    optimizerG, optimizerD = None, None
    criterionD = None
    if opt.cs:
        print("Creating GAN for confusing samples")
        netG = Generator(num_points=opt.num_points).cuda()
        netD = Discriminator().cuda()
        criterionD = nn.BCELoss()
        # move to distributed
        netG = DDP(netG, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
        netD = DDP(netD, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
        optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.cs_gan_lr, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.cs_gan_lr, betas=(0.5, 0.999))

    start_epoch = 1
    glob_it = 0
    if opt.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ckt = torch.load(opt.resume, map_location=map_location)
        model.load_state_dict(ckt['model'], strict=True)
        if opt.script_mode != 'train_exposure':
            # resume experiment
            optimizer.load_state_dict(ckt['optimizer'])
            scheduler.load_state_dict(ckt['scheduler'])
            if opt.cs:
                netG.load_state_dict(ckt['netG'])
                netD.load_state_dict(ckt['netD'])
            if scaler is not None:
                assert 'scaler' in ckt.keys(), "No scaler key in ckt"
                assert ckt['scaler'] is not None, "None scaler object in ckt"
                scaler.load_state_dict(ckt['scaler'])
            if rank == 0:
                logger.cprint("Restart training from checkpoint %s" % opt.resume)
            start_epoch += int(ckt['epoch'])
            glob_it += (int(ckt['epoch']) * len(train_loader))
        else:
            # load model weights for OE finetuning
            print(f"Finetuning model {opt.resume} for outlier exposure")
        del ckt

    # TRAINER
    opt.glob_it = glob_it  # will be update by the train_epoch fun.
    opt.gan_glob_it = glob_it
    best_epoch, best_acc = -1, -1
    time1 = time.time()
    for epoch in range(start_epoch, opt.epochs + 1):
        is_best = False
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        if opt.script_mode == 'train_exposure':
            # finetuning clf for Outlier Exposure with mixup data
            train_epoch_rsmix_exposure(epoch=epoch, args=opt, train_loader=train_loader, model=model, scaler=scaler,
                                       optimizer=optimizer, logger=logger)
        else:
            # training clf from scratch
            if opt.cs:
                # train gan for ARPL
                train_epoch_cs(epoch=epoch, args=opt, train_loader=train_loader, model=model, netD=netD, netG=netG,
                               scaler=scaler, optimizer=optimizer, criterionD=criterionD, optimizerD=optimizerD,
                               optimizerG=optimizerG, logger=logger)

            train_epoch_cla(epoch=epoch, args=opt, train_loader=train_loader, model=model, scaler=scaler,
                            optimizer=optimizer, logger=logger)

        # step lr
        scheduler.step(epoch)

        # evaluation for classification
        if epoch % opt.eval_step == 0:
            _, src_pred, src_labels = get_confidence(model, test_loader)
            src_pred = to_numpy(src_pred)
            src_labels = to_numpy(src_labels)
            epoch_acc = accuracy_score(src_labels, src_pred)
            epoch_bal_acc = balanced_accuracy_score(src_labels, src_pred)
            if rank == 0:
                logger.cprint(f"Test [{epoch}/{opt.epochs}]\tAcc: {epoch_acc:.4f}, Bal Acc: {epoch_bal_acc:.4f}")
                wandb.log({"test/ep_acc": epoch_acc, "test/ep_bal_acc": epoch_bal_acc, "test/epoch": epoch})
                is_best = epoch_acc >= best_acc
                if is_best:
                    best_acc = epoch_acc
                    best_epoch = epoch

        # save checkpoint
        if rank == 0:
            ckt_path = osp.join(opt.models_dir, "model_last.pth")
            save_checkpoint(opt, ckt_path, model, optimizer, scheduler, scaler, config, epoch)
            if is_best:
                os.system('cp -r {} {}'.format(ckt_path, osp.join(opt.models_dir, f"model_best.pth")))
            if epoch % opt.save_step == 0:
                os.system('cp -r {} {}'.format(ckt_path, osp.join(opt.models_dir, f"model_ep{epoch}.pth")))

    train_time = time.time() - time1
    if rank == 0:
        logger.cprint(f"Training finished - best test acc: {best_acc:.4f} at ep.: {best_epoch}, time: {train_time}")


def eval_r2r(opt, config):
    set_random_seed(opt.seed)

    print(f"Arguments: {opt}")
    train_loader, val_loader, id_loader, ood_loader = get_loaders_test(opt)
    classes_dict = eval(opt.src)
    n_classes = len(set(classes_dict.values()))
    model = Classifier(args=DotConfig(config['model']), num_classes=n_classes, loss=opt.loss, cs=opt.cs)
    ckt_weights = torch.load(opt.ckpt_path, map_location='cpu')['model']
    ckt_weights = sanitize_model_dict(ckt_weights)
    ckt_weights = convert_model_state(ckt_weights, model.state_dict())
    print(f"Model params count: {count_parameters(model) / 1000000 :.4f} M")
    print("Load weights: ", model.load_state_dict(ckt_weights, strict=True))
    model = model.cuda().eval()

    src_logits, src_pred, src_labels = get_network_output(model, id_loader)
    tar_logits, _, _ = get_network_output(model, ood_loader)

    # compute test accuracy
    src_labels = to_numpy(src_labels)
    src_pred = to_numpy(src_pred)
    src_acc = skm.accuracy_score(src_labels, src_pred)
    src_bal_acc = skm.balanced_accuracy_score(src_labels, src_pred)
    print(f"Src Test - Acc: {src_acc}, Bal Acc: {src_bal_acc}\n")

    # MSP
    print("\n" + "#" * 80)
    print("Computing OOD metrics with MSP normality score...")
    src_MSP_scores = F.softmax(src_logits, dim=1).max(1)[0]
    tar_MSP_scores = F.softmax(tar_logits, dim=1).max(1)[0]
    res = get_ood_metrics(src_MSP_scores, tar_MSP_scores, src_label=1)
    print(res)
    print("#" * 80)

    # MLS
    print("\n" + "#" * 80)
    print("Computing OOD metrics with MLS normality score...")
    src_MLS_scores = src_logits.max(1)[0]
    tar_MLS_scores = tar_logits.max(1)[0]
    res = get_ood_metrics(src_MLS_scores, tar_MLS_scores, src_label=1)
    print(res)
    print("#" * 80)

    # entropy
    print("\n" + "#" * 80)
    src_entropy_scores = 1 / logits_entropy_loss(src_logits)
    tar_entropy_scores = 1 / logits_entropy_loss(tar_logits)
    print("Computing OOD metrics with entropy normality score...")
    res = get_ood_metrics(src_entropy_scores, tar_entropy_scores, src_label=1)
    print(res)
    print("#" * 80)

    # FEATURES EVALUATION
    eval_OOD_with_feats(model, train_loader, id_loader, ood_loader, save_feats=opt.save_feats)

    # ODIN
    print("\n" + "#" * 80)
    print("Computing OOD metrics with ODIN normality score...")
    src_odin = iterate_data_odin(model, id_loader)
    tar_odin = iterate_data_odin(model, ood_loader)
    res = get_ood_metrics(src_odin, tar_odin, src_label=1)
    print(res)
    print("#" * 80)

    # Energy
    print("\n" + "#" * 80)
    print("Computing OOD metrics with Energy normality score...")
    src_energy = iterate_data_energy(model, id_loader)
    tar_energy = iterate_data_energy(model, ood_loader)
    res = get_ood_metrics(src_energy, tar_energy, src_label=1)
    print(res)
    print("#" * 80)

    # GradNorm
    print("\n" + "#" * 80)
    print("Computing OOD metrics with GradNorm normality score...")
    src_gradnorm = iterate_data_gradnorm(model, id_loader)
    tar_gradnorm = iterate_data_gradnorm(model, ood_loader)
    res = get_ood_metrics(src_gradnorm, tar_gradnorm, src_label=1)
    print(res)
    print("#" * 80)

    # React with id-dependent threshold
    print("\n" + "#" * 80)
    threshold = estimate_react_thres(model, val_loader)
    print(f"Computing OOD metrics with React (+Energy) normality score, ID-dependent threshold (={threshold:.4f})...")
    print(f"React - using {opt.src} test to compute threshold")
    src_react = iterate_data_react(model, id_loader, threshold=threshold)
    tar_react = iterate_data_react(model, ood_loader, threshold=threshold)
    res = get_ood_metrics(src_react, tar_react, src_label=1)
    print(res)
    print("#" * 80)
    return



# slightly modified for real->real scenario
def eval_OOD_with_feats(model, train_loader, src_loader, tar_loader, tar2_loader=None, save_feats=None):
    from knn_cuda import KNN
    knn = KNN(k=1, transpose_mode=True)

    print("\n" + "#" * 80)
    print("Computing OOD metrics with distance from train features...")
    # extract penultimate features, compute distances
    train_feats, train_labels = get_penultimate_feats(model, train_loader)
    src_feats, src_labels = get_penultimate_feats(model, src_loader)
    tar_feats, tar_labels = get_penultimate_feats(model, tar_loader)
    if tar2_loader:
        tar2_feats, tar2_labels = get_penultimate_feats(model, tar2_loader)
    else:
        tar2_feats, tar2_labels = None, None

    train_labels = train_labels.cpu().numpy()
    labels_set = set(train_labels)
    prototypes = torch.zeros((len(labels_set), train_feats.shape[1]), device=train_feats.device)
    for idx, lbl in enumerate(labels_set):
        mask = train_labels == lbl
        prototype = train_feats[mask].mean(0)
        prototypes[idx] = prototype

    if save_feats is not None:
        if isinstance(train_loader.dataset, ModelNet40_OOD):
            labels_2_names = {v: k for k, v in train_loader.dataset.class_choice.items()}
        else:
            labels_2_names = {}

        output_dict = {}
        output_dict["labels_2_names"] = labels_2_names
        output_dict["train_feats"], output_dict["train_labels"] = train_feats.cpu(), train_labels
        output_dict["id_data_feats"], output_dict["id_data_labels"] = src_feats.cpu(), src_labels
        output_dict["ood1_data_feats"], output_dict["ood1_data_labels"] = tar_feats.cpu(), tar_labels
        if tar2_loader:
            output_dict["ood2_data_feats"], output_dict["ood2_data_labels"] = tar2_feats.cpu(), tar2_labels
        torch.save(output_dict, save_feats)
        print(f"Features saved to {save_feats}")


    ################################################
    print("Euclidean distances in a non-normalized space:")
    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(train_feats.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = 1 / src_dist
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    # OOD tar1
    tar1_dist, _ = knn(train_feats.unsqueeze(0), tar_feats.unsqueeze(0))
    tar1_dist = tar1_dist.squeeze().cpu()
    tar1_scores = 1 / tar1_dist

    if tar2_loader:
        # OOD tar2
        tar2_dist, _ = knn(train_feats.unsqueeze(0), tar2_feats.unsqueeze(0))
        tar2_dist = tar2_dist.squeeze().cpu()
        tar2_scores = 1 / tar2_dist
        eval_ood_sncore(
            scores_list=[src_scores, tar1_scores, tar2_scores],
            preds_list=[src_pred, None, None],  # [src_pred, None, None],
            labels_list=[src_labels, None, None],  # [src_labels, None, None],
            src_label=1  # confidence should be higher for ID samples
        )
    else:
        src_labels = to_numpy(src_labels)
        src_pred = to_numpy(src_pred)
        src_acc = skm.accuracy_score(src_labels, src_pred)
        src_bal_acc = skm.balanced_accuracy_score(src_labels, src_pred)
        print(f"Src Test - Acc: {src_acc}, Bal Acc: {src_bal_acc}\n")
        ood_res = get_ood_metrics(src_scores, tar1_scores, src_label=1)
        print(ood_res)


    ################################################
    print("\nEuclidean distances with prototypes:")
    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(prototypes.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = 1 / src_dist
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    # OOD tar1
    tar1_dist, _ = knn(prototypes.unsqueeze(0), tar_feats.unsqueeze(0))
    tar1_dist = tar1_dist.squeeze().cpu()
    tar1_scores = 1 / tar1_dist

    if tar2_loader:
        # OOD tar2
        tar2_dist, _ = knn(prototypes.unsqueeze(0), tar2_feats.unsqueeze(0))
        tar2_dist = tar2_dist.squeeze().cpu()
        tar2_scores = 1 / tar2_dist
        eval_ood_sncore(
            scores_list=[src_scores, tar1_scores, tar2_scores],
            preds_list=[src_pred, None, None],
            labels_list=[src_labels, None, None],
            src_label=1  # confidence should be higher for ID samples
        )
    else:
        src_labels = to_numpy(src_labels)
        src_pred = to_numpy(src_pred)
        src_acc = skm.accuracy_score(src_labels, src_pred)
        src_bal_acc = skm.balanced_accuracy_score(src_labels, src_pred)
        print(f"Src Test - Acc: {src_acc}, Bal Acc: {src_bal_acc}\n")
        ood_res = get_ood_metrics(src_scores, tar1_scores, src_label=1)
        print(ood_res)
    print("#" * 80)


    ################################################
    print("\nCosine similarities on the hypersphere:")
    # cosine sim in a normalized space
    train_feats = F.normalize(train_feats, p=2, dim=1)
    src_feats = F.normalize(src_feats, p=2, dim=1)
    tar_feats = F.normalize(tar_feats, p=2, dim=1)
    src_scores, src_ids = torch.mm(src_feats, train_feats.t()).max(1)
    tar1_scores, _ = torch.mm(tar_feats, train_feats.t()).max(1)
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    if tar2_loader:
        tar2_feats = F.normalize(tar2_feats, p=2, dim=1)
        tar2_scores, _ = torch.mm(tar2_feats, train_feats.t()).max(1)
        eval_ood_sncore(
            scores_list=[(0.5 * src_scores + 0.5).cpu(), (0.5 * tar1_scores + 0.5).cpu(), (0.5 * tar2_scores + 0.5).cpu()],
            preds_list=[src_pred, None, None],  # [src_pred, None, None],
            labels_list=[src_labels, None, None],  # [src_labels, None, None],
            src_label=1  # confidence should be higher for ID samples
        )
    else:
        src_labels = to_numpy(src_labels)
        src_pred = to_numpy(src_pred)
        src_acc = skm.accuracy_score(src_labels, src_pred)
        src_bal_acc = skm.balanced_accuracy_score(src_labels, src_pred)
        print(f"Src Test - Acc: {src_acc}, Bal Acc: {src_bal_acc}\n")
        ood_res = get_ood_metrics((0.5 * src_scores + 0.5).cpu(), (0.5 * tar1_scores + 0.5).cpu(), src_label=1)
        print(ood_res)
    print("#" * 80)


    ################################################
    print("\nCosine similarities with prototypes:")
    # cosine sim in a normalized space
    prototypes = F.normalize(prototypes, p=2, dim=1)
    src_feats = F.normalize(src_feats, p=2, dim=1)
    tar_feats = F.normalize(tar_feats, p=2, dim=1)
    src_scores, src_ids = torch.mm(src_feats, prototypes.t()).max(1)
    tar1_scores, _ = torch.mm(tar_feats, prototypes.t()).max(1)
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    if tar2_loader:
        tar2_feats = F.normalize(tar2_feats, p=2, dim=1)
        tar2_scores, _ = torch.mm(tar2_feats, prototypes.t()).max(1)
        eval_ood_sncore(
            scores_list=[(0.5 * src_scores + 0.5).cpu(), (0.5 * tar1_scores + 0.5).cpu(), (0.5 * tar2_scores + 0.5).cpu()],
            preds_list=[src_pred, None, None],
            labels_list=[src_labels, None, None],
            src_label=1  # confidence should be higher for ID samples
        )
    else:
        src_labels = to_numpy(src_labels)
        src_pred = to_numpy(src_pred)
        src_acc = skm.accuracy_score(src_labels, src_pred)
        src_bal_acc = skm.balanced_accuracy_score(src_labels, src_pred)
        print(f"Src Test - Acc: {src_acc}, Bal Acc: {src_bal_acc}\n")
        ood_res = get_ood_metrics((0.5 * src_scores + 0.5).cpu(), (0.5 * tar1_scores + 0.5).cpu(), src_label=1)
        print(ood_res)
    print("#" * 80)


def main():
    args = get_args()
    config = load_yaml(args.config)

    if args.script_mode.startswith('train'):
        # launch trainer
        print("training...")
        assert args.checkpoints_dir is not None and len(args.checkpoints_dir)
        assert args.exp_name is not None and len(args.exp_name)
        args.log_dir = osp.join(args.checkpoints_dir, args.exp_name)
        args.tb_dir = osp.join(args.checkpoints_dir, args.exp_name, "tb-logs")
        args.models_dir = osp.join(args.checkpoints_dir, args.exp_name, "models")
        args.backup_dir = osp.join(args.checkpoints_dir, args.exp_name, "backup-code")
        train(args, config)
    else:
        # eval real -> real
        assert args.ckpt_path is not None and len(args.ckpt_path)
        print("out-of-distribution eval - real -> real ..")
        eval_r2r(args, config)


if __name__ == '__main__':
    main()
