#!/bin/bash


python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=False -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_flip' -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='v_flip' -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=10 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=20 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=30 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

wait

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=40 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=50 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=60 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=70 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=80 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &

python3 augment.py -totEpochs=40 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=90 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/' &
