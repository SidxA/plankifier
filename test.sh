##!/bin/bash

python3 optimizer.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt='adam_2' -datapath='./data/' -loss='categorical_crossentropy' &
python3 optimizer.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=False -resize=keep_proportions -bs=8 -lr=0.0001 -opt='adam_2' -datapath='./data/' -loss='categorical_crossentropy' &

wait

python3 optimizer.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=True -resize=keep_proportions -bs=8 -lr=0.0001 -opt='sgd_2' -datapath='./data/' -loss='categorical_crossentropy' &
python3 optimizer.py -totEpochs=100 -width=128 -height=128 -model=conv2 -aug=False -resize=keep_proportions -bs=8 -lr=0.0001 -opt='sgd_2' -datapath='./data/' -loss='categorical_crossentropy' &
