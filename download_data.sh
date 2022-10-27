#!/bin/bash
data_dir=$(pwd)/3D_OS_release_data  # change here data_dir if required
mkdir -p "$data_dir"  # make dir if it does not exists
echo "Downloading data in ${data_dir}"


# sncore
printf "============"
echo "Downloading ShapeNetCore resampled in $dir"
sncore="https://www.dropbox.com/s/oa3qbujpugw4d43/sncore_fps_4096.tar?dl=1"
wget -O "${data_dir}/tmp_sncore_fps_4096.tar" "$sncore"  # download
tar -xf "${data_dir}/tmp_sncore_fps_4096.tar" -C "$data_dir"  # extract
rm "${data_dir}/tmp_sncore_fps_4096.tar" # clean
printf "============\n"


# md40 + ood splits
printf "============"
echo "Downloading ModelNet40 + OOD Splits in $dir"
md40="https://www.dropbox.com/s/c2x3h59nxprjs21/modelnet40_normal_resampled.tar?dl=1"
wget -O "${data_dir}/tmp_modelnet40_normal_resampled.tar" "$md40"  # download
tar -xf "${data_dir}/tmp_modelnet40_normal_resampled.tar" -C "$data_dir"  # extract
rm "${data_dir}/tmp_modelnet40_normal_resampled.tar" # clean
printf "============\n"


# scanobject
printf "============"
echo "Downloading ScanObjectNN in $dir"
sonn="https://www.dropbox.com/s/gu0p3rych1k26b7/ScanObjectNN.tar?dl=1"
wget -O "${data_dir}/tmp_ScanObjectNN.tar" "$sonn"  # download
tar -xf "${data_dir}/tmp_ScanObjectNN.tar" -C "$data_dir"  # extract
rm "${data_dir}/tmp_ScanObjectNN.tar" # clean
printf "============\n"


# md40 corruputed
printf "============"
echo "Downloading ModelNet40 with corruptions in $dir"
md40_corruptions="https://www.dropbox.com/s/28u4swbyyn3wflz/ModelNet40_corrupted.tar?dl=1"
wget -O "${data_dir}/tmp_ModelNet40_corrupted.tar" "$md40_corruptions"  # download
tar -xf "${data_dir}/tmp_ModelNet40_corrupted.tar" -C "$data_dir"  # extract
rm "${data_dir}/tmp_ModelNet40_corrupted.tar" # clean
printf "============\n"


echo "Finished"
