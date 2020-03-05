#!/bin/bash


python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=False -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=True -augtype='brightness' -augparameter=0.1 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=True -augtype='rescale' -augparameter=1.5 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

wait

python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=True -augtype='brightness' -augparameter=0.3 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=1 -width=128 -height=128 -model=conv2 -aug=True -augtype='rescale' -augparameter=15 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &
