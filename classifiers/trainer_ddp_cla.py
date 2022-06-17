import sys
import os
import os.path as osp

sys.path.append(os.getcwd())
import time
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import *
from utils.dist import *
from utils.data_utils import *
from datasets.sncore_4k import ShapeNetCore4k
# noinspection PyUnresolvedReferences
from datasets.sncore_splits import *
# noinspection PyUnresolvedReferences
from models.classifiers import Classifier
from utils.ood_utils import get_confidence, eval_ood_sncore, iterate_data_odin, \
    iterate_data_energy, iterate_data_gradnorm, iterate_data_react, estimate_react_thres, print_ood_output, \
    get_penultimate_feats, get_network_output
import wandb
from base_args import add_base_args
from models.common import convert_model_state, logits_entropy_loss
from models.ARPL_utils import Generator, Discriminator
from classifiers.common import train_epoch_rsmix_exposure, train_epoch_cs, train_epoch_cla



def get_args():
    parser = argparse.ArgumentParser("OOD on point clouds via contrastive learning")
    parser = add_base_args(parser)
    # experiment specific args
    parser.add_argument("--augm_set",
                        type=str, default="st", help="data augm - st is only scale+translate",
                        choices=["st", "rw"])
    parser.add_argument("--grad_norm_clip",
                        default=-1, type=float, help="gradient clipping")
    parser.add_argument("--num_points",
                        default=1024, type=int, help="number of points sampled for each object view")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_proj", type=str, default="benchmark-3d-ood-cla")
    parser.add_argument("--cs", action='store_true', help="Enable confusing samples for ARPL")
    parser.add_argument("--cs_gan_lr", type=float, default=0.0002, help="Confusing samples GAN lr")
    parser.add_argument("--cs_beta", type=float, default=0.1, help="Beta loss weight for CS")
    parser.add_argument("--loss", type=str, default="CE",
                        choices=["CE", "CE_ls", "cosface", "arcface", "subcenter_arcface", "ARPL", "cosine"],
                        help="Which loss to use for training. CE is default")
    parser.add_argument("--save_feats", type=str, default=None,
                        help="Path where to save feats of penultimate layer")
    parser.add_argument("--exposure_head_only", action='store_true', help="Finetune head only in exposure training")

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


def get_sncore_train_loader(opt, synset=None, split="train"):
    world_size = get_ws()
    rank = get_rank()
    drop_last = not str(opt.script_mode).startswith('eval')
    if opt.augm_set == 'st':
        print("Augm set ST")
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),
            AugmScale(lo=2 / 3, hi=3 / 2),
            AugmTranslate(translate_range=0.2)]
    elif opt.augm_set == 'rw':
        print("Augm set RW")
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),
            AugmScale(),
            AugmRotate(axis=[0.0, 1.0, 0.0]),
            AugmRotatePerturbation(),
            AugmTranslate(),
            AugmJitter()]
    else:
        raise ValueError(f"Unknown augmentation set: {opt.augm_set}")

    print(f"Train transforms: {set_transforms}")
    train_transforms = transforms.Compose(set_transforms)
    if synset is None:
        synset = opt.src

    train_data = ShapeNetCore4k(
        data_root=opt.data_root, split=split, class_choice=list(eval(synset).keys()),
        num_points=4096, transforms=train_transforms, apply_fix_cellphone=opt.apply_fix_cellphone)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
                              sampler=train_sampler, worker_init_fn=init_np_seed)
    return train_loader


def get_sncore_val_loader(opt):
    # Also compatible with DDP training runtime evaluation
    ws = get_ws()
    rank = get_rank()
    drop_last = not str(opt.script_mode).startswith('eval')

    base_data_params = {
        'data_root': opt.data_root, 'split': "val", 'num_points': opt.num_points, 'transforms': None,
        'apply_fix_cellphone': opt.apply_fix_cellphone}

    val_data = ShapeNetCore4k(**base_data_params, class_choice=list(eval(opt.src).keys()))
    val_sampler = DistributedSampler(val_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    print(f"val_sampler: {val_sampler}")
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=6,
                            sampler=val_sampler, worker_init_fn=init_np_seed)
    return val_loader


def get_test_loaders(opt):

    """
    Returns all dataloaders used for evaluation, this function is compatible with DDP training runtime evaluation

    Return values:
        train_loader: train loader to compute category centroids - no augm, shuffle=True, drop_last=True if Training else False
        test_loader: test ID data loader - no augm, shuffle=True, drop_last=True if Training else False
        tar1_loader: test OOD 1 data loader - no augm, shuffle=True, drop_last=True if Training else False
        tar2_loader: test OOD 2 data loader - no augm, shuffle=True, drop_last=True if Training else False

    """

    ws, rank = get_ws(), get_rank()
    drop_last = not str(opt.script_mode).startswith('eval')

    base_data_params = {
        'data_root': opt.data_root, 'num_points': opt.num_points, 'transforms': None,
        'apply_fix_cellphone': opt.apply_fix_cellphone}

    print(f"OOD evaluation data - "
          f"src: {opt.src}, tar1: {opt.tar1}, tar2: {opt.tar2}")

    # in domain training data
    train_data = ShapeNetCore4k(**base_data_params, split='train', class_choice=list(eval(opt.src).keys()))
    # in domain test data
    test_data = ShapeNetCore4k(**base_data_params, split='test', class_choice=list(eval(opt.src).keys()))
    # targets (out of domain data) test data
    tar1_data = ShapeNetCore4k(**base_data_params, split='test', class_choice=list(eval(opt.tar1).keys()))
    tar2_data = ShapeNetCore4k(**base_data_params, split='test', class_choice=list(eval(opt.tar2).keys()))

    # samplers - are None if not distributed training
    train_sampler = DistributedSampler(train_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    test_sampler = DistributedSampler(test_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    tar1_sampler = DistributedSampler(tar1_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None
    tar2_sampler = DistributedSampler(tar2_data, num_replicas=ws, rank=rank, shuffle=True) if is_dist() else None

    # loaders
    train_loader = DataLoader( # train loader with no augmentation for feature eval
        train_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
        sampler=train_sampler, worker_init_fn=init_np_seed)
    test_loader = DataLoader(
        test_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
        sampler=test_sampler, worker_init_fn=init_np_seed)
    tar1_loader = DataLoader(
        tar1_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
        sampler=tar1_sampler, worker_init_fn=init_np_seed)
    tar2_loader = DataLoader(
        tar2_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
        sampler=tar2_sampler, worker_init_fn=init_np_seed)

    return test_loader, tar1_loader, tar2_loader, train_loader


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

    # get training loader
    train_loader = get_sncore_train_loader(opt)

    if isinstance(train_loader.dataset, ShapeNetCore4k) and rank == 0:
        # logs class mapping
        logger.cprint(f"train - id_2_label: {str(train_loader.dataset.id_2_label)}\n")

    # dataloaders for evaluation
    src_loader, tar1_loader, tar2_loader, _ = get_test_loaders(opt)

    # build model
    n_classes = train_loader.dataset.num_classes
    print(f"Number of training classes: {n_classes}")
    model = Classifier(args=DotConfig(config['model']), num_classes=n_classes, loss=opt.loss, cs=opt.cs)
    if str(config['model']['ENCO_NAME']).lower() == "gdanet":
        model.apply(weight_init_GDA)
    else:
        model.apply(weights_init_normal)

    # move to CUDA, build DDP
    model = model.cuda()
    if opt.use_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if rank == 0:
        logger.cprint(f"model: \n{model}")
        logger.cprint(f"Model params count: {count_parameters(model) / 1000000:.4f} M")
        logger.cprint(f"loss: {opt.loss}\n")

    model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    if rank == 0:
        wandb.watch(model, log="gradients")

    # optimizer and scheduler
    named_parameters = model.named_parameters()
    if opt.script_mode == 'train_exposure' and opt.exposure_head_only:
        named_parameters = model.module.head.named_parameters()

    optimizer, scheduler = get_opti_sched(named_parameters, config)
    scaler = GradScaler(enabled=opt.use_amp)

    netG, netD = None, None
    optimizerG, optimizerD = None, None
    criterionD = None
    if opt.cs:
        print("Creating GAN for confusing samples")
        netG = Generator(num_points=opt.num_points).cuda()
        netD = Discriminator().cuda()
        criterionD = nn.BCELoss()
        netG = DDP(netG, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
        netD = DDP(netD, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
        optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.cs_gan_lr, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.cs_gan_lr, betas=(0.5, 0.999))

    start_epoch = 1
    glob_it = 0
    if opt.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # configure map_location properly
        ckt = torch.load(opt.resume, map_location=map_location)
        model.load_state_dict(ckt['model'], strict=True)
        if opt.script_mode != 'train_exposure':
            # restart training from checkpoint
            # loads also optimizer and scheduler
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
            # finetune model weights for Outlier Exposure
            if rank == 0:
                logger.cprint(f"Finetuning model {opt.resume} for outlier exposure")
        del ckt

    # TRAINER
    opt.glob_it = glob_it
    opt.gan_glob_it = glob_it
    best_epoch, best_acc = -1, -1
    time1 = time.time()
    for epoch in range(start_epoch, opt.epochs + 1):
        is_best = False

        if isinstance(train_loader, DataLoader) and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if opt.script_mode == 'train_exposure':
            train_epoch_rsmix_exposure(
                epoch=epoch,
                args=opt,
                train_loader=train_loader,
                model=model,
                scaler=scaler,
                optimizer=optimizer,
                logger=logger)
        else:
            if opt.cs:
                # train gan
                train_epoch_cs(epoch=epoch, args=opt, train_loader=train_loader, model=model, netD=netD, netG=netG,
                               scaler=scaler, optimizer=optimizer, criterionD=criterionD, optimizerD=optimizerD,
                               optimizerG=optimizerG, logger=logger)

            # train one epoch
            train_epoch_cla(epoch=epoch, args=opt, train_loader=train_loader, model=model, scaler=scaler,
                            optimizer=optimizer, logger=logger)

        # step lr
        scheduler.step(epoch)

        # evaluation routine
        if epoch % opt.eval_step == 0:
            src_loader.sampler.set_epoch(epoch)
            tar1_loader.sampler.set_epoch(epoch)
            tar2_loader.sampler.set_epoch(epoch)
            start_eval = time.time()
            # MSP as validation metric
            src_conf, src_pred, src_labels = get_confidence(model, src_loader)
            tar1_conf, _, _ = get_confidence(model, tar1_loader)
            tar2_conf, _, _ = get_confidence(model, tar2_loader)
            src_acc, src_bal_acc, _, _, res_big_tar = eval_ood_sncore(
                scores_list=[src_conf, tar1_conf, tar2_conf],
                preds_list=[src_pred, None, None],
                labels_list=[src_labels, None, None],
                src_label=1,
                silent=True
            )
            # report auroc src -> tar1+tar2
            auroc, fpr = res_big_tar['auroc'], res_big_tar['fpr_at_95_tpr']
            eval_time = time.time() - start_eval
            is_best = src_acc >= best_acc
            if is_best:
                best_acc = max(src_acc, best_acc)
                best_epoch = epoch
            if rank == 0:
                wandb.log({"test/acc": src_acc, "test/bal_acc": src_bal_acc,
                           "test/auroc": auroc, "test/fpr": fpr, "test/epoch": epoch})
                logger.cprint(f"Acc: {src_acc}, Bal Acc: {src_bal_acc}, AUROC: {auroc:.4f}, FPR: {fpr:.4f}")
                logger.cprint(f"eval time: {eval_time}\n")

        # save checkpoint
        if rank == 0:
            # save last
            ckt_path = osp.join(opt.models_dir, "model_last.pth")
            save_checkpoint(opt, ckt_path, model, optimizer, scheduler, scaler, config, epoch, netG=netG, netD=netD)
            if is_best:
                os.system('cp -r {} {}'.format(ckt_path, osp.join(opt.models_dir, f"model_best.pth")))
            if epoch % opt.save_step == 0:
                os.system('cp -r {} {}'.format(ckt_path, osp.join(opt.models_dir, f"model_ep{epoch}.pth")))

    time2 = time.time()
    if rank == 0:
        logger.cprint(
            f"Training finished - best test acc: {best_acc:.4f} at epoch: {best_epoch}, time: {time2 - time1}")


def eval_ood(opt, config):
    print(f"Arguments: {opt}")
    set_random_seed(opt.seed)

    src_loader, tar1_loader, tar2_loader, train_loader = get_test_loaders(opt)
    n_classes = src_loader.dataset.num_classes
    model = Classifier(args=DotConfig(config['model']), num_classes=n_classes, loss=opt.loss, cs=opt.cs)
    ckt_weights = torch.load(opt.ckpt_path, map_location='cpu')['model']
    ckt_weights = sanitize_model_dict(ckt_weights)
    ckt_weights = convert_model_state(ckt_weights, model.state_dict())
    print(f"Model params count: {count_parameters(model) / 1000000:.4f} M")
    print("Load weights: ", model.load_state_dict(ckt_weights, strict=True))
    model = model.cuda().eval()

    # MSP
    print("\n" + "#" * 80)
    print("Computing OOD metrics with MSP normality score...")

    src_logits, src_pred, src_labels = get_network_output(model, src_loader)
    tar1_logits, _, _ = get_network_output(model, tar1_loader)
    tar2_logits, _, _ = get_network_output(model, tar2_loader)

    src_MSP_scores = F.softmax(src_logits, dim=1).max(1)[0]
    tar1_MSP_scores = F.softmax(tar1_logits, dim=1).max(1)[0]
    tar2_MSP_scores = F.softmax(tar2_logits, dim=1).max(1)[0]

    eval_ood_sncore(
        scores_list=[src_MSP_scores, tar1_MSP_scores, tar2_MSP_scores],
        preds_list=[src_pred, None, None],  # computes also MSP accuracy on ID test set
        labels_list=[src_labels, None, None],  # computes also MSP accuracy on ID test set
        src_label=1
    )
    print("#" * 80)

    src_MLS_scores = src_logits.max(1)[0]
    tar1_MLS_scores = tar1_logits.max(1)[0]
    tar2_MLS_scores = tar2_logits.max(1)[0]

    # MLS
    print("\n" + "#" * 80)
    print("Computing OOD metrics with MLS normality score...")
    eval_ood_sncore(
        scores_list=[src_MLS_scores, tar1_MLS_scores, tar2_MLS_scores],
        preds_list=[src_pred, None, None],  # computes also MSP accuracy on ID test set
        labels_list=[src_labels, None, None],  # computes also MSP accuracy on ID test set
        src_label=1
    )
    print("#" * 80)

    src_entropy_scores = 1 / logits_entropy_loss(src_logits)
    tar1_entropy_scores = 1 / logits_entropy_loss(tar1_logits)
    tar2_entropy_scores = 1 / logits_entropy_loss(tar2_logits)

    # entropy
    print("\n" + "#" * 80)
    print("Computing OOD metrics with entropy normality score...")
    eval_ood_sncore(
        scores_list=[src_entropy_scores, tar1_entropy_scores, tar2_entropy_scores],
        preds_list=[src_pred, None, None],  # computes also MSP accuracy on ID test set
        labels_list=[src_labels, None, None],  # computes also MSP accuracy on ID test set
        src_label=1
    )
    print("#" * 80)

    if opt.loss in ["ARPL", "cosine"]:
        return

    # FEATURES EVALUATION
    eval_OOD_with_feats(model, train_loader, src_loader, tar1_loader, tar2_loader, save_feats=opt.save_feats)

    # ODIN
    print("\n" + "#" * 80)
    print("Computing OOD metrics with ODIN normality score...")
    src_odin = iterate_data_odin(model, src_loader)
    tar1_odin = iterate_data_odin(model, tar1_loader)
    tar2_odin = iterate_data_odin(model, tar2_loader)
    eval_ood_sncore(scores_list=[src_odin, tar1_odin, tar2_odin], src_label=1)
    print("#" * 80)

    # Energy
    print("\n" + "#" * 80)
    print("Computing OOD metrics with Energy normality score...")
    src_energy = iterate_data_energy(model, src_loader)
    tar1_energy = iterate_data_energy(model, tar1_loader)
    tar2_energy = iterate_data_energy(model, tar2_loader)
    eval_ood_sncore(scores_list=[src_energy, tar1_energy, tar2_energy], src_label=1)
    print("#" * 80)

    # GradNorm
    print("\n" + "#" * 80)
    print("Computing OOD metrics with GradNorm normality score...")
    src_gradnorm = iterate_data_gradnorm(model, src_loader)
    tar1_gradnorm = iterate_data_gradnorm(model, tar1_loader)
    tar2_gradnorm = iterate_data_gradnorm(model, tar2_loader)
    eval_ood_sncore(scores_list=[src_gradnorm, tar1_gradnorm, tar2_gradnorm], src_label=1)
    print("#" * 80)

    # React with id-dependent threshold
    print("\n" + "#" * 80)
    val_loader = get_sncore_val_loader(opt)
    threshold = estimate_react_thres(model, val_loader)
    print(f"Computing OOD metrics with React (+Energy) normality score, ID-dependent threshold (={threshold:.4f})...")
    src_react = iterate_data_react(model, src_loader, threshold=threshold)
    tar1_react = iterate_data_react(model, tar1_loader, threshold=threshold)
    tar2_react = iterate_data_react(model, tar2_loader, threshold=threshold)
    eval_ood_sncore(scores_list=[src_react, tar1_react, tar2_react], src_label=1)
    print("#" * 80)
    return


def eval_OOD_with_feats(model, train_loader, src_loader, tar1_loader, tar2_loader, save_feats=None):
    from knn_cuda import KNN
    knn = KNN(k=1, transpose_mode=True)

    print("\n" + "#" * 80)
    print("Computing OOD metrics with distance from train features...")

    # extract penultimate features, compute distances
    train_feats, train_labels = get_penultimate_feats(model, train_loader)
    src_feats, src_labels = get_penultimate_feats(model, src_loader)
    tar1_feats, tar1_labels = get_penultimate_feats(model, tar1_loader)
    tar2_feats, tar2_labels = get_penultimate_feats(model, tar2_loader)
    train_labels = train_labels.cpu().numpy()

    labels_set = set(train_labels)
    prototypes = torch.zeros((len(labels_set), train_feats.shape[1]), device=train_feats.device)
    for idx, lbl in enumerate(labels_set):
        mask = train_labels == lbl
        prototype = train_feats[mask].mean(0)
        prototypes[idx] = prototype

    if save_feats is not None:
        if isinstance(train_loader.dataset, ShapeNetCore4k):
            labels_2_sids = {v: k for k, v in train_loader.dataset.id_2_label.items()}
            sids_2_names = sncore_all_synset
            labels_2_names = {lbl: sids_2_names[labels_2_sids[lbl]] for lbl in labels_set}
        else:
            labels_2_names = {}

        output_dict = {}
        output_dict["labels_2_names"] = labels_2_names
        output_dict["train_feats"], output_dict["train_labels"] = train_feats.cpu(), train_labels
        output_dict["id_data_feats"], output_dict["id_data_labels"] = src_feats.cpu(), src_labels
        output_dict["ood1_data_feats"], output_dict["ood1_data_labels"] = tar1_feats.cpu(), tar1_labels
        output_dict["ood2_data_feats"], output_dict["ood2_data_labels"] = tar2_feats.cpu(), tar2_labels
        torch.save(output_dict, save_feats)
        print(f"Features saved to {save_feats}")

    ################################################
    print("\nEuclidean distances in a non-normalized space:")
    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(train_feats.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = 1 / src_dist
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

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
        preds_list=[src_pred, None, None],
        labels_list=[src_labels, None, None],
        src_label=1)

    print("\nEuclidean distances with prototypes:")
    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(prototypes.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = 1 / src_dist
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    # OOD tar1
    tar1_dist, _ = knn(prototypes.unsqueeze(0), tar1_feats.unsqueeze(0))
    tar1_dist = tar1_dist.squeeze().cpu()
    tar1_scores = 1 / tar1_dist

    # OOD tar2
    tar2_dist, _ = knn(prototypes.unsqueeze(0), tar2_feats.unsqueeze(0))
    tar2_dist = tar2_dist.squeeze().cpu()
    tar2_scores = 1 / tar2_dist

    eval_ood_sncore(
        scores_list=[src_scores, tar1_scores, tar2_scores],
        preds_list=[src_pred, None, None],
        labels_list=[src_labels, None, None],
        src_label=1)

    ################################################
    print("\nCosine similarities on the hypersphere:")
    # cosine sim in a normalized space
    train_feats = F.normalize(train_feats, p=2, dim=1)
    src_feats = F.normalize(src_feats, p=2, dim=1)
    tar1_feats = F.normalize(tar1_feats, p=2, dim=1)
    tar2_feats = F.normalize(tar2_feats, p=2, dim=1)
    src_scores, src_ids = torch.mm(src_feats, train_feats.t()).max(1)
    tar1_scores, _ = torch.mm(tar1_feats, train_feats.t()).max(1)
    tar2_scores, _ = torch.mm(tar2_feats, train_feats.t()).max(1)
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample
    eval_ood_sncore(
        scores_list=[(0.5 * src_scores + 0.5).cpu(), (0.5 * tar1_scores + 0.5).cpu(), (0.5 * tar2_scores + 0.5).cpu()],
        preds_list=[src_pred, None, None],
        labels_list=[src_labels, None, None],
        src_label=1)

    print("\nCosine similarities with prototypes:")
    # cosine sim in a normalized space
    prototypes = F.normalize(prototypes, p=2, dim=1)
    src_feats = F.normalize(src_feats, p=2, dim=1)
    tar1_feats = F.normalize(tar1_feats, p=2, dim=1)
    tar2_feats = F.normalize(tar2_feats, p=2, dim=1)
    src_scores, src_ids = torch.mm(src_feats, prototypes.t()).max(1)
    tar1_scores, _ = torch.mm(tar1_feats, prototypes.t()).max(1)
    tar2_scores, _ = torch.mm(tar2_feats, prototypes.t()).max(1)
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample
    eval_ood_sncore(
        scores_list=[(0.5 * src_scores + 0.5).cpu(), (0.5 * tar1_scores + 0.5).cpu(), (0.5 * tar2_scores + 0.5).cpu()],
        preds_list=[src_pred, None, None],
        labels_list=[src_labels, None, None],
        src_label=1)
    print("#" * 80)


def main():
    args = get_args()
    assert args.config is not None and osp.exists(args.config)
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
    elif args.script_mode == 'eval':
        print("out-of-distribution evaluation...")
        assert args.ckpt_path is not None and len(args.ckpt_path)
        eval_ood(args, config)
    else:
        raise ValueError(f"Unknown script mode: {args.script_mode}")


if __name__ == '__main__':
    main()
