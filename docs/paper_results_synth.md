# Synthetic Benchmark 

## Discriminative methods

```bash
# train
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE 
# eval
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE \
    -mode eval --ckpt_path outputs/DGCNN_CE_SN1/models/model_last.pth
```

The eval outputs performance results for all of the discriminative methods normality scores (**MSP**,
**MLS**, **ODIN**, **Energy**, **GradNorm** and **ReAct**). 

This should be repeated for the three sets (SN1, SN2, SN3) and for both backbones, by using
`--config cfgs/dgcnn-cla.yaml` for DGCNN and `--config cfgs/pn2-msg.yaml` for PointNet++. 

## Density and Reconstructon Based Models

**VAE**

TODO

**NF**

```bash
# train
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_NF.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_NF_SN1 --src SN1
# eval
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_NF.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_NF_SN1 --src SN1 \
    -mode eval --ckpt_path outputs/DGCNN_NF_SN1/models/model_last.pth
```

This should be repeated for the three sets (SN1, SN2, SN3) and for both backbones, by using
`--config cfgs/dgcnn-cla.yaml` for DGCNN and `--config cfgs/pn2-msg.yaml` for PointNet++. 
The output results with the loglikelihood normality score are those reported in the paper. 

## Outlier Exposure with OOD Generated Data

These experiments perform a finetuning over a model trained for classification. Thus it's necessary
to first of all perform a training like done for discriminative models: 

```bash
# first training step for classification
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE 
# outlier exposure finetuning
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn_exposure.yaml --exp_name DGCNN_OE_SN1 --src SN1 --loss CE \
    --resume outputs/DGCNN_CE_SN1/models/model_last.pth --epochs 100 -mode 'train_exposure'
# eval
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_OE_SN1 --src SN1 --loss CE \
    -mode eval --ckpt_path outputs/DGCNN_OE_SN1/models/model_last.pth
```

This should be repeated for the three sets (SN1, SN2, SN3) and for both backbones, by using:

 - `--config cfgs/dgcnn-cla.yaml` for DGCNN pretraining and `--config/dgcnn_exposure.yaml` for DGCNN finetuning;
 - `--config cfgs/pn2-msg.yaml` for PointNet++ pretraining, `--config/pn2-msg_exposure.yaml` for PointNet++ finetuning. 

The interesting result, which we reported in the paper, corresponds with the output MSP metric. 

## Representation and Distance Based Models

**ARPL+CS**

```bash
# train
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_ARPL_CS_SN1 --src SN1 --loss ARPL --cs
# eval
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_ARPL_CS_SN1 --src SN1 --loss ARPL --cs \
    -mode eval --ckpt_path outputs/DGCNN_ARPL_CS_SN1/models/model_last.pth
```

This should be repeated for the three sets (SN1, SN2, SN3) and for both backbones, by using
`--config cfgs/dgcnn-cla.yaml` for DGCNN and `--config cfgs/pn2-msg.yaml` for PointNet++. 
The interesting result, which we reported in the paper, corresponds with the output MLS metric. 

**Cosine proto**

```bash
# train
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_cosine_SN1 --src SN1 --loss cosine
# eval
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_cosine_SN1 --src SN1 --loss cosine \
    -mode eval --ckpt_path outputs/DGCNN_cosine_SN1/models/model_last.pth
```

This should be repeated for the three sets (SN1, SN2, SN3) and for both backbones, by using
`--config cfgs/dgcnn-cla.yaml` for DGCNN and `--config cfgs/pn2-msg.yaml` for PointNet++. 
The interesting result, which we reported in the paper, corresponds with the output MLS metric. 

**CE(L2)**

We use the same training and eval procedure of the discriminative methods. At eval time the
interesting output result is the one dubbed *Euclidean distances in a non-normalized space*.

**SupCon**

```bash
# train
python -m torch.distributed.launch --nproc_per_node=1 supcon/trainer_ddp_supcon.py --config cfgs/dgcnn_contrast.yaml --exp_name DGCNN_supcon_SN1 --src SN1
# eval 
python -m torch.distributed.launch --nproc_per_node=1 supcon/trainer_ddp_supcon.py --config cfgs/dgcnn_contrast.yaml --exp_name DGCNN_supcon_SN1 --src SN1 \
    -mode eval --ckpt_path outputs/DGCNN_supcon_SN1/models/model_last.pth
```

This should be repeated for the three sets (SN1, SN2, SN3) and for both backbones, by using
`--config cfgs/dgcnn_contrast.yaml` for DGCNN and `--config cfgs/pn2-msg_contrast.yaml` for PointNet++. 
The output results reported in the paper are those called *Evaluation - Nearest Training Sample*.


**SubArcFace**

```bash
# train
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/face_losses_SN/dgcnn_face_SN.yaml --exp_name DGCNN_subarcface_SN1 --src SN1 --loss subcenter_arcface --epochs 500
# eval
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/face_losses_SN/dgcnn_face_SN.yaml --exp_name DGCNN_subarcface_SN1 --src SN1 --loss subcenter_arcface --epochs 500 \
    -mode eval --ckpt_path outputs/DGCNN_subarcface_SN1/models/model_last.pth
```

This should be repeated for the three sets (SN1, SN2, SN3) and for both backbones, by using
`--config cfgs/face_losses_SN/dgcnn_face_SN.yaml` for DGCNN and `--config cfgs/face_losses_SN/pn2-msg_face_SN.yaml` for PointNet++. 
The output results reported in the paper are those called *Cosine similarities on the hypersphere*.

