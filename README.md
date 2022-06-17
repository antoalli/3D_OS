# Towards Open Set 3D Learning: Benchmarking and Understanding Semantic Novelty Detection on Point Clouds

Official PyTorch implementation of  "Towards Open Set 3D Learning: Benchmarking
and Understanding Semantic Novelty Detection on Point Clouds" by Antonio Alliegro, Francesco Cappio
Borlino and Tatiana Tommasi. 

> **Abstract:** *In recent years there has been significant progress in the field of 3D learning on classification, detection and segmentation problems. The vast majority of the existing studies focus on canonical closed-set conditions, neglecting the intrinsic open nature of the real-world. This limits the abilities of robots and autonomous systems involved in safety-critical applications that require managing novel and unknown signals. In this context exploiting 3D data can be a valuable asset since it provides rich information about the geometry of sensed objects and scenes. With this paper we provide a first broad study on open set 3D learning. We introduce a novel testbed for semantic novelty detection that considers several settings with increasing difficulties in terms of category semantic shift, and covers both in-domain (synthetic-to-synthetic) and cross-domain (synthetic-to-real) scenarios. Moreover, we investigate the related open set 2D literature to understand if and how its recent improvements are effective on 3D data. Our extensive benchmark positions several algorithms in the same coherent picture, revealing their strengths and limitations. The results of our analysis may serve as a reliable foothold for future tailored open set 3D models.*

## Introduction

This code allows to replicate all the experiments and reproduce all the results that we included in
our paper.

### Requirements
We perform our experiments with PyTorch 1.9.1+cu111 and Python 3.7. To install all the required packages simply run:

```bash
pip install -r requirements.txt
```

**Additional libraries** 

N.B. to install PointNet++ ops the system-wide CUDA version must match the PyTorch one (CUDA 11 in this case).

```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"
```

### Data

TODO

## Run example

### Training

Each experiment requires to choose a backbone (through the config file), a loss function and a source set. For example: 

```bash
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE 
```

Details: 

 - the training seed can be specified via `--seed <int>` argument. Seeds used for our paper experiments
   are: 1, 41, 13718. 
 - the source set determines which class set will be used for *known* classes in the given experiment. In case of synth->synth experiments the other two sets 
   among [SN1, SN2, SN3] will define *unknown* classes. In case of synth->real experiments the
   *unknown* classes set is composed of the other among [SR1, SR2] and a common OOD set. In practice
   in each experiments there are two different sets of unknown classes.
 - training output is stored in `outputs/<exp_name>`. 

### Eval 

```bash
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_ddp_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE \
    -mode eval --ckpt_path outputs/DGCNN_CE_SN1/models/model_last.pth
```

Example output:
```bash
Computing OOD metrics with MLS normality score...
AUROC - Src label: 1, Tar label: 0
Src Test - Clf Acc: 0.8522321428571429, Clf Bal Acc: 0.7607892805466309
Auroc 1: 0.7345, FPR 1: 0.7905
Auroc 2: 0.7574, FPR 2: 0.7458
Auroc 3: 0.7440, FPR 3: 0.7719
to spreadsheet: 0.734540224202969,0.7904599659284497,0.7573800391095067,0.7457882069795427,0.7440065016031351,0.7719451371571072
```

The output contains closed set accuracy on known classes (Clf Acc), balanced closed set accuracy
(Clf Bal Acc) and 3 sets of open set performance results. In the paper we report AUROC 3 and FPR 3
which refer to the scenario `(known) src -> unknown set 1 + unknown set 2`.

## Replicating paper results

In the following we report the commands necessary to replicate all of the main paper results (Table
1 and 3). All experiments of the paper are repeated with the 3 seeds specified above and the results
is averaged across the three runs.

See [here](docs/paper_results_synth.md) for Synthetic Benchmark results replication and [here](docs/paper_results_real.md) for Synthetic to Real Benchmark. 


## This repo is still under construction
- [x] Upload Code
- [ ] Upload Data
- [ ] Acknowledgements
- [ ] Link to the arXiv and citation
