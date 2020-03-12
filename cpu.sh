##!/bin/bash


python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='categorical_crossentropy' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='mean_squared_error' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='mean_absolute_error' &

wait

python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='mean_absolute_percentage_error' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='mean_squared_logarithmic_error' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='squared_hinge' &

wait

python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='logcosh' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='huber_loss' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='binary_crossentropy' &

wait

python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='kullback_leibler_divergence' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='poisson' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='cosine_proximity' &

wait

python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='categorical_hinge' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='is_categorical_crossentropy' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='sparse_categorical_crossentropy' &

wait

python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='hinge' &
python3 loss.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt=adam -datapath='./data/' -loss='categorical_hinge' &
