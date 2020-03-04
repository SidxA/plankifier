#!/bin/bash
#augmented reality

#run the script for rotation, angle goes from 0 to 360 degrees
#for i in {30,60,90,120,150,180,210,240,270,300,330,360}:
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='rotate' -augparameter=i -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'

#run the script for w_shift, fraction of total width
#for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}:
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='w_shift' -augparameter=i -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'
	
#run the script for h_shift, fraction of total width
#for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}:
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_shift' -augparameter=i -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'
	
#run the script for shear, angle goes from 0 to 90
#for i in {5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95}:
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=i -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'
	
#run the script for zoom
#for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}:
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='zoom' -augparameter=i -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'
	
#run the script for horizontal flip
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='h_flip' -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'
	
#run the script for vertical flip
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='v_flip' -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'
	
#run the script for brightness
#for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}:
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='brightness' -augparameter=(0,i) -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'
	
#run the script for rescale
#for i in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}:
#	python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='rescale' -augparameter=i -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='./data/'


python augment.py -totEpochs=10 -width=128 -height=128 -model=conv2 -aug=True -augtype='shear' -augparameter=45 -resize=keep_proportions -bs=8 -lr=0.0001 -opt=sgd -datapath='~/data/'
