#!/bin/bash


python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' -augparameter=$360 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' & >> A.txt

python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=45 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' & >> B.txt

python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=90 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' & >> C.txt
