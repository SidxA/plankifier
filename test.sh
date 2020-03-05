#!/bin/bash

for i in {0..40..10}
do
python3 augment.py -totEpochs=50 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' -augparameter=$i -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'
done
