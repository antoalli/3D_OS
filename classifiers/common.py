import time
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
import wandb
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils.utils import cal_ce_loss
from utils.rsmix_provider import rsmix
from utils.dist import *
from utils.utils import AverageMeter, to_numpy


def train_epoch_cla(epoch, args, train_loader, model, scaler, optimizer, logger):
    train_preds = []
    train_targets = []
    rank = get_rank()
    ws = get_ws()
    time1 = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    model.train()

    for i, batch in enumerate(train_loader, 0):
        optimizer.zero_grad(set_to_none=True)
        args.glob_it += 1
        curr_lr = optimizer.param_groups[0]['lr']
        data_time.update(time.time() - end)

        points = batch[0].cuda(non_blocking=True)  # [bs,num_points,3]
        targets = batch[1].cuda(non_blocking=True).long()  # [bs,]
        bs = len(targets)  # num shapes in minibatch

        """ Single optimization step """
        with autocast(enabled=args.use_amp):
            if str(args.loss).startswith("CE"):
                logits = model(points)
                loss = cal_ce_loss(pred=logits, target=targets, smoothing=(args.loss == "CE_ls"))
            elif args.loss == "ARPL":
                logits, loss = model(points, targets)
            else:
                # cosface, arcface, ecc
                train_logits, logits = model(points, targets)
                loss = F.cross_entropy(train_logits, targets)

        scaler.scale(loss).backward()

        if args.grad_norm_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm_clip, norm_type=2)

        scaler.step(optimizer)
        scaler.update()

        preds = to_numpy(logits.max(dim=1)[1])
        targets = to_numpy(targets)
        train_preds.append(preds)
        train_targets.append(targets)

        # logging
        losses.update(loss.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 10 == 0:
            log_str = f"it: [{i + 1}/{len(train_loader)}-{epoch}/{args.epochs}], rank: [{rank + 1}/{ws}], " \
                      f"Loss: {losses.val:.4f}, Loss avg: {losses.avg:.4f}, lr: {curr_lr:.6f}, " \
                      f"BT: {batch_time.val:.2f}, DT: {data_time.val:.2f}"
            if rank == 0:
                logger.cprint(log_str)
                res_it = {"train/it_loss": loss.item(), "train/it_lr": curr_lr, 'train/it': args.glob_it}
                wandb.log(res_it)
            else:
                print(log_str)

    time2 = time.time()
    epoch_loss = losses.avg
    train_targets = np.concatenate(train_targets, 0)
    train_preds = np.concatenate(train_preds, 0)
    epoch_acc = accuracy_score(train_targets, train_preds)
    epoch_bal_acc = balanced_accuracy_score(train_targets, train_preds)

    log_str = f"Train [{epoch}/{args.epochs}]\t" \
              f"rank: [{rank + 1}/{ws}], " \
              f"Loss: {epoch_loss:.4f}, " \
              f"Acc: {epoch_acc:.4f}, " \
              f"Bal Acc: {epoch_bal_acc:.4f}, " \
              f"BT: {batch_time.avg:.2f}, " \
              f"DT: {data_time.avg:.2f},  " \
              f"epoch time: {(time2 - time1):.2f}"

    res_epoch = {"train/ep_loss": epoch_loss, "train/ep_acc": epoch_acc, "train/ep_bal_acc": epoch_bal_acc,
                 "train/epoch": epoch, "train/ep_time": (time2 - time1)}
    if rank == 0:
        logger.cprint(log_str)
        wandb.log(res_epoch)
    else:
        print(log_str)

    return res_epoch


def train_epoch_rsmix_exposure(epoch, args, train_loader, model, scaler, optimizer, logger):
    """
    Executes a single finetuning epoch for Outlier Exposure
    OOD data samples are produced as Rigid Sample Mix (RSMix, Lee et al CVPR 21) of ID data samples
    """
    train_preds = []
    train_targets = []
    rank = get_rank()
    ws = get_ws()
    time1 = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    exp_loss_meter = AverageMeter()
    end = time.time()
    assert args.resume is not None, "model ckt to finetune (--resume) is needed"
    assert args.loss == "CE"
    mixup_nsample = 512
    model.train()

    for i, batch in enumerate(train_loader, 0):
        optimizer.zero_grad(set_to_none=True)
        args.glob_it += 1
        curr_lr = optimizer.param_groups[0]['lr']
        data_time.update(time.time() - end)

        points = batch[0].numpy()  # [bs, num_points, 3]
        targets = batch[1].numpy()  # [bs,]
        bs = len(targets)
        if rank == 0 and i == 0:
            logger.cprint(f"targets: {targets}")

        # obtain mixup pointclouds
        rsmix_points, lam, label, label_b = rsmix(
            points, targets, beta=1.0, n_sample=mixup_nsample, KNN=False)

        # to tensor
        points = torch.as_tensor(points)  # [bs,num_points,3]
        rsmix_points = torch.as_tensor(rsmix_points)
        targets = torch.as_tensor(targets).cuda(non_blocking=True)  # [bs,]
        # first 'bs' are ID, second 'bs' are rs_mixup
        all_points = torch.cat([points, rsmix_points], 0).cuda(non_blocking=True).float()  # [bs * 2, num_points, 3]

        with autocast(enabled=args.use_amp):
            logits = model(all_points)
            # cross-entropy on ID (not mixed) samples
            cls_loss = F.cross_entropy(logits[:bs], targets)

            # cross-entropy from softmax distribution to uniform distribution
            # see https://github.com/hendrycks/outlier-exposure/issues/19#issuecomment-855714395
            exp_loss = 0.5 * -(logits[bs:].mean(1) - torch.logsumexp(logits[bs:], dim=1)).mean()

            loss = cls_loss + exp_loss

        scaler.scale(loss).backward()

        if args.grad_norm_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm_clip, norm_type=2)

        scaler.step(optimizer)
        scaler.update()

        # accuracy ID
        preds = to_numpy(logits[:bs].max(dim=1)[1])
        targets = to_numpy(targets)
        train_preds.append(preds)
        train_targets.append(targets)

        # logging
        loss_meter.update(loss.item(), bs)
        cls_loss_meter.update(cls_loss.item(), bs)
        exp_loss_meter.update(exp_loss.item(), bs)
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 10 == 0:
            log_str = f"it: [{i + 1}/{len(train_loader)}-{epoch}/{args.epochs}], rank: [{rank + 1}/{ws}], " \
                      f"Tot Loss: {loss_meter.val:.4f}, Tot Loss avg: {loss_meter.avg:.4f}, " \
                      f"Cls Loss: {cls_loss_meter.val:.4f}, Cls Loss avg: {cls_loss_meter.avg:.4f}, " \
                      f"Exp Loss: {exp_loss_meter.val:.4f}, Exp Loss avg: {exp_loss_meter.avg:.4f}, " \
                      f"lr: {curr_lr:.6f}, BT: {batch_time.val:.2f}, DT: {data_time.val:.2f}"
            if rank == 0:
                logger.cprint(log_str)
                res_it = {"train/it_loss": loss.item(), "train/it_lr": curr_lr, 'train/it': args.glob_it,
                          "train/it_cls_loss": cls_loss.item(), "train/it_exposure_loss": exp_loss.item()}
                wandb.log(res_it)
            else:
                print(log_str)

    time2 = time.time()
    epoch_loss = loss_meter.avg
    train_targets = np.concatenate(train_targets, 0)
    train_preds = np.concatenate(train_preds, 0)
    epoch_acc = accuracy_score(train_targets, train_preds)
    epoch_bal_acc = balanced_accuracy_score(train_targets, train_preds)

    log_str = f"Train [{epoch}/{args.epochs}]\t" \
              f"rank: [{rank + 1}/{ws}], " \
              f"Loss: {epoch_loss:.4f}, " \
              f"Acc: {epoch_acc:.4f}, " \
              f"Bal Acc: {epoch_bal_acc:.4f}, " \
              f"BT: {batch_time.avg:.2f}, " \
              f"DT: {data_time.avg:.2f},  " \
              f"epoch time: {(time2 - time1):.2f}"

    res_epoch = {"train/ep_loss": epoch_loss, "train/ep_acc": epoch_acc, "train/ep_bal_acc": epoch_bal_acc,
                 "train/epoch": epoch}

    if rank == 0:
        logger.cprint(log_str)
        wandb.log(res_epoch)
    else:
        print(log_str)

    return res_epoch


def train_epoch_cs(epoch, args, train_loader, model, netD, netG, scaler, optimizer, criterionD, optimizerD, optimizerG, logger):
    """
    Train one epoch of the GAN
    """

    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()

    rank = get_rank()
    ws = get_ws()
    time1 = time.time()
    model.train()
    netG.train()
    netD.train()

    torch.cuda.empty_cache()

    # size of noise used for input to the generator
    z_dim = 96

    gan_real_label = 0
    gan_fake_label = 1

    if args.grad_norm_clip > 0:
        # grad clipping
        raise NotImplementedError

    for i, batch in enumerate(train_loader, 0):
        args.gan_glob_it += 1

        points = batch[0].cuda(non_blocking=True)  # [bs,num_points,3]
        targets = batch[1].cuda(non_blocking=True)  # [bs,]
        bs = len(targets)

        # generate fake data
        noise = torch.FloatTensor(bs, z_dim).normal_(0, 1).cuda()
        fake = netG(noise)

        ##################################
        # Update Discriminator Network   #
        ##################################
        optimizerD.zero_grad()

        with autocast(enabled=args.use_amp):
            # train with real
            output = netD(points)
            gan_target = gan_real_label * torch.ones((targets.shape[0], 1)).cuda()
            errD_real = criterionD(output, gan_target)

            # train with fake
            output = netD(fake)
            gan_target = gan_fake_label * torch.ones((targets.shape[0], 1)).cuda()
            errD_fake = criterionD(output, gan_target)

            # total
            errD = errD_real + errD_fake

        scaler.scale(errD).backward(retain_graph=True)
        scaler.step(optimizerD)

        lossesD.update(errD.item(), bs)

        ##################################
        # Update Generator Network       #
        ##################################
        optimizerG.zero_grad()

        with autocast(enabled=args.use_amp):
            # gan loss: the generator should aim at confusing the discriminator
            gan_target = gan_real_label * torch.ones((targets.shape[0], 1)).cuda()
            output = netD(fake)
            errG = criterionD(output, gan_target)

            # minimize generated samples distances from reciprocal points
            # via entropy maximization
            errG_F = model(fake, bn_label=gan_fake_label, fake_loss=True).mean()

            generator_loss = errG + args.cs_beta * errG_F

        scaler.scale(generator_loss).backward(retain_graph=True)
        scaler.step(optimizerG)

        lossesG.update(generator_loss.item(), bs)

        ##################################
        # Update Classifier Network      #
        ##################################
        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            # cross entropy loss on real data
            _, loss = model(points, targets, bn_label=gan_real_label)

        scaler.scale(loss).backward(retain_graph=True)

        with autocast(enabled=args.use_amp):
            # maximize entropy loss on fake data
            noise = torch.FloatTensor(bs, z_dim).normal_(0, 1).cuda()
            fake = netG(noise)
            fake_loss = model(fake, bn_label=gan_fake_label, fake_loss=True).mean()

        total_loss = loss + args.cs_beta * fake_loss

        scaler.scale(fake_loss * args.cs_beta).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(total_loss.item(), bs)

        # logging
        if (i + 1) % 10 == 0:
            log_str = f"GAN it: [{i + 1}/{len(train_loader)}-{epoch}/{args.epochs}], rank: [{rank + 1}/{ws}], " \
                      f"Loss: {losses.val:.4f}, Loss avg: {losses.avg:.4f}, " \
                      f"LossD: {lossesD.val:.4f}, LossD avg: {lossesD.avg:.4f}, " \
                      f"LossG: {lossesG.val:.4f}, LossG avg: {lossesG.avg:.4f} "
            if rank == 0:
                logger.cprint(log_str)
                res_it = {
                    "train/GAN/D_loss": lossesD.val, "train/GAN/G_loss": lossesG.val,
                    "train/GAN/C_loss": losses.val, "train/GAN/it": args.gan_glob_it
                }
                wandb.log(res_it)
            else:
                print(log_str)

    time2 = time.time()

    log_str = f"Train [{epoch}/{args.epochs}]\t" \
              f"rank: [{rank + 1}/{ws}], " \
              f"Loss: {losses.avg:.4f}, " \
              f"LossD: {lossesD.avg:.4f}, " \
              f"LossG: {lossesG.avg:.4f}, " \
              f"epoch time: {(time2 - time1):.2f}"

    res_epoch = {
        "train/GAN/ep_loss": losses.avg,
        "train/GAN/ep_lossD": lossesD.avg,
        "train/GAN/ep_lossG": lossesG.avg,
        "train/GAN/epoch": epoch}

    if rank == 0:
        logger.cprint(log_str)
        wandb.log(res_epoch)
    else:
        print(log_str)

    return res_epoch
