#!/bin/bash
dir=$(pwd)  # change here datasets root if needed
mkdir -p "$dir"
echo "Downloading data in $dir"

# download data
wget -O "${dir}/3D_OS_release_data.tar" "https://www.dropbox.com/s/2ta5o5sa4q69fsb/3D_OS_release_data.tar?dl=1"
# modelnet
echo "Downloading modelnet in $dir"
wget --no-check-certificate -O "${dir}/modelnet40_normal_resampled.zip" "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"

# extract data
tar -xf "${dir}/3D_OS_release_data.tar" -C "$dir"
# move modelnet zip to 3D_OS_release_data and extract
mv "${dir}/modelnet40_normal_resampled.zip" "${dir}/3D_OS_release_data/"
unzip -d "${dir}/3D_OS_release_data/" "${dir}/3D_OS_release_data/modelnet40_normal_resampled.zip"

# clean
echo "Cleaning.."
rm "${dir}/3D_OS_release_data/modelnet40_normal_resampled.zip"
rm "${dir}/3D_OS_release_data.tar"
echo "Finished"
