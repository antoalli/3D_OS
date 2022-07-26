# Towards Open Set 3D Learning: A Benchmark on Object Point Clouds

Official PyTorch implementation of ["Towards Open Set 3D Learning: A Benchmark on Object Point Clouds"](https://arxiv.org/abs/2207.11554) by Antonio Alliegro, Francesco Cappio
Borlino and Tatiana Tommasi. 

> **Abstract:** *In the last years, there has been significant progress in the field of 3D learning on classification, detection and segmentation problems. The vast majority of the existing studies focus on canonical closed-set conditions, neglecting the intrinsic open nature of the real-world. This limits the abilities of autonomous systems involved in safety-critical applications that require managing novel and unknown signals. In this context exploiting 3D data can be a valuable asset since it conveys rich information about the geometry of sensed objects and scenes. This paper provides the first broad study on Open Set 3D learning. We introduce a novel testbed with settings of increasing difficulty in terms of category semantic shift and cover both in-domain (synthetic-to-synthetic) and cross-domain (synthetic-to-real) scenarios. Moreover, we investigate the related out-of-distribution and Open Set 2D literature to understand if and how their most recent approaches are effective on 3D data. Our extensive benchmark positions several algorithms in the same coherent picture, revealing their strengths and limitations. The results of our analysis may serve as a reliable foothold for future tailored Open Set 3D models.*

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
Use the prepared script to download all datasets. 

```bash
chmod +x download_data.sh
./download_data.sh
```

A common root for all datasets will be created in project dir, by default named *3D_OS_release_data* .
```
3D_OS_release_data (root)
├─ ModelNet40_corrupted
├─ sncore_fps_4096
├─ ScanObjectNN
├─ modelnet40_normal_resampled
```

The absolute path to the datasets root must be passed as **--data_root** argument in all scripts.


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

## Citation 

If you find our paper/code useful please consider citing our paper: 

```
@article{alliegro3dos,
  title={Towards Open Set 3D Learning: A Benchmark on Object Point Clouds},
  author={Alliegro, Antonio and Cappio Borlino, Francesco and Tommasi, Tatiana},
  journal={arXiv preprint arXiv:2207.11554},
  year={2022}
}
```


## This repo is still under construction
- [x] Upload Code
- [x] Upload Data
- [ ] Acknowledgements
- [x] Link to the arXiv and citation
