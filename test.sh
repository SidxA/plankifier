##!/bin/bash


#wait till a possible prior run is finished (in seconds)

wait 

python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=2 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=4 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=16 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=24 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=32 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=64 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=96 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=128 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=48 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=80 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 batchsize.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=112 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.1 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.2 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.3 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.4 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.5 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.6 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.7 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.8 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' =augparameter=0.9 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=30 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=60 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=90 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=120 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=150 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=180 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=210 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=240 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=270 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=300 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=330 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' =augparameter=360 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.1 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.2 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.3 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.4 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.5 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.6 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.7 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.8 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=3 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' =augparameter=0.9 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.1 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.2 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.3 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.4 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.5 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.6 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.7 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.8 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' =augparameter=0.9 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.1 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.2 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.3 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &

wait

python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.4 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.5 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.6 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.7 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.8 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
python3 augment.py -totEpochs=2 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' =augparameter=0.9 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' &
