#!/bin/bash
git submodule init && git submodule update
if [ ! -d models ]; then
  mkdir models
fi
if [ ! -f models/d2_ots.pth ]; then
  wget https://dsmn.ml/files/d2-net/d2_ots.pth -O models/d2_ots.pth
fi
if [ ! -f models/d2_tf.pth ]; then
  wget https://dsmn.ml/files/d2-net/d2_tf.pth -O models/d2_tf.pth
fi
if [ ! -f models/d2_tf_no_phototourism.pth ]; then
  wget https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth -O models/d2_tf_no_phototourism.pth
fi
if [ ! -d test_images ]; then
  mkdir test_images
fi
if [ ! -d test_images/same_things ]; then
  mkdir test_images/same_things
fi
if [ ! -d test_images/mount_rushmore ]; then
  wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/mount_rushmore.tar.gz -P test_images/
  tar -xvf test_images/mount_rushmore.tar.gz -C test_images/same_things
  rm test_images/mount_rushmore.tar.gz
fi
echo "Setup done"
