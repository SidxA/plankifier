#!/usr/bin/env python3
#
# run as python3 features.py
#
#########################################################################


import pandas as pd, numpy as np, os
from PIL import Image
import os, sys, pathlib, time, datetime, argparse, numpy as np, pandas as pd
import keras as k
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, concatenate
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_models, helper_data
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser(description='Train image classifier cosisting of a cnn and a parallel mlp for the features files')
parser.add_argument('-datapath', default='./small_data/', help="Data directory")
#parser.add_argument('-datakind', default='image', choices=['mixed','image','tsv'], help="If tsv, expect a single tsv file; if images, each class directory has only images inside; if mixed, expect a more complicated structure defined by the output of SPCConvert")
parser.add_argument('-outpath', default='./out/', help="Print many messages on screen.")
parser.add_argument('-verbose', default= 1, help="one of [0,1,2] for amount of output training documentation")
#parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")
parser.add_argument('-totEpochs', type=int, default=10, help="Total number of epochs for the training")
parser.add_argument('-opt', default='sgd_1', help="Choice of the minimization algorithm")
parser.add_argument('-bs', type=int, default=8, help="Batch size")
parser.add_argument('-lr', type=float, default=0.0001, help="Learning Rate")
parser.add_argument('-height', type=int, default=128, help="Image height")
parser.add_argument('-width', type=int, default=128, help="Image width")
parser.add_argument('-depth', type=int, default=3, help="Number of channels")
parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
parser.add_argument('-aug', default = True, help="Perform data augmentation.")
#parser.add_argument('-resize', choices=['keep_proportions','acazzo'], default='keep_proportions', help='The way images are resized')
#parser.add_argument('-model', choices=['mlp','conv2','smallvgg'], default='conv2', help='The model. MLP gives decent results, conv2 is the best, smallvgg overfits (*validation* accuracy oscillates).')
#parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
#parser.add_argument('-load', default=None, help='Path to a previously trained model that should be loaded.')
#parser.add_argument('-override_lr', action='store_true', help='If true, when loading a previously trained model it discards its LR in favor of args.lr')
#parser.add_argument('-initial_epoch', type=int, default=0, help='Initial epoch of the training')
parser.add_argument('-augtype', default='standard', help='Augmentation type')
parser.add_argument('-augparameter', type=float, default=0, help='Augmentation parameter')
parser.add_argument('-cpu', default=False, help='performs training only on cpus')
parser.add_argument('-gpu', default=False, help='performs training only on gpus')
args=parser.parse_args()

if args.gpu:
	from tensorflow.compat.v1 import ConfigProto
	from tensorflow.compat.v1 import InteractiveSession

	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
	
if args.cpu:
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	
	

imagesize = args.height
datapath = args.datapath
learning_rate = args.lr
batchsize=args.bs
epochs=args.totEpochs

#THE COMBINED INPUT OF PICTURE DATA AND FEATURE DATA

#output directory
namestring = 'mixed_'
'''
if args.cpu:
	namestring += 'cpu_'
if args.gpu:
	namestring += 'gpu_'
if args.opt != 'sgd_1':
	namestring += np.str(args.opt) +'_'
if args.totEpochs != 10:
	namestring += np.str(args.totEpochs) + 'epoch(s)_'
if args.bs != 8:
	namestring += 'bs:' + np.str(args.bs) + '_'
if args.lr != 0.0001:
	namestring += 'lr:' + np.str(args.lr) +'_'
if args.testSplit != 0.2:
	namestring += 'split:' + np.str(args.testSplit) + '_'
if args.height != 128:
	namestring += 'imagesize_on_' + np.str(args.width)+'_'
if args.aug == False:
	namestring += 'noaug_'
if args.augtype != 'standard':
	namestring += 'aug:' + np.str(args.augtype) + '_on_' + np.str(args.augparameter) + '_'
'''
now = datetime.datetime.now()
dt_string = now.strftime("%Y-%m-%d_%Hh%Mm%Ss")
filename = namestring + dt_string
outDir = args.outpath+'/'+filename+'/'
pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)
fsummary=open(outDir+'args.txt','w')
print(args, file=fsummary); fsummary.flush()

#READING IN THE DATASET
def dataintake(datapath, imagesize):
    
    df = pd.DataFrame(columns=['name', 'npimage', 'rescaled', 'label','feature'])
    
    classes = {'name': [ name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name)) ]}
    classes['num']    = len(classes['name'])
    classes['num_ex'] =  np.zeros(classes['num'], dtype=int)
    labels = []
    
    for ic in range(classes['num']):
        c=classes['name'][ic]
        features = pd.read_csv(datapath+c+'/features.tsv', sep = '\t')
        classPath=datapath+c+'/'
        classImages = os.listdir(classPath)
        #throwing out the first and second row which are date and image size
        features.drop(features.columns[[1,2]], axis=1, inplace=True)
        features.set_index('url', inplace = True)
        ima = os.listdir(datapath+c+'/training_data')
        #print(features.index)
        for imagename in ima:
            imagePath = datapath+c+'/training_data/'+imagename
            image = Image.open(imagePath)
            image,rescaled = helper_data.ResizeWithProportions(image, imagesize) # width and height are assumed to be the same (assertion at the beginning)
            npimage = np.array(image.copy() , dtype=np.float32)/255.0 #rescale
            
            #only takes the images where there are features for
            
            if 'images/00000/'+imagename in features.index:
                df=df.append({'name':imagePath, 'npimage':npimage, 'rescaled':rescaled, 'label':ic, 'feature':np.array(features.loc['images/00000/'+imagename,:])}, ignore_index=True)
            #else:
            #    df=df.append({'name':imagePath, 'npimage':npimage, 'rescaled':rescaled, 'label':ic}, ignore_index=True)
            image.close()
            
            labels=np.concatenate(( labels, np.full(classes['num_ex'][ic], ic) ), axis=0)
    classes['tot_ex'] =  classes['num_ex'].sum()
    labels = np.array(labels)
    #np.save(outDir+'classes.npy', classes)
            
    return df, labels, classes
df, labels, classes = dataintake(datapath,imagesize)

#no test implemented yet
x1 = np.stack(df['npimage'])#.reshape(imagesize,imagesize,3,df['npimage'].shape[0])
x2 = np.stack(df['feature'])
y = np.stack(df['label'])
#(trainX, testX, trainY, testY) = train_test_split((df['npimage'],df['feature']),df['label'],test_size=0.2)
#train_size=len(trainX)
#test_size=len(testX)
#lb = LabelBinarizer()
#trainY = lb.fit_transform(trainY)
#testY = lb.transform(testY)

y = k.utils.to_categorical(y, num_classes=None, dtype='float32')

######
#MODEL CREATION
######
featurenumber = df['feature'][0].shape[0]
classnumber = classes['num']

I1 = k.layers.Input(shape=(imagesize,imagesize,3), name = 'Image_Input')#Image shape
I2 = k.layers.Conv2D(filters = 8, kernel_size = 3, name = 'Image_conv1')(I1)#64,3,relu
I3 = k.layers.Conv2D(filters = 8, kernel_size = 3, name = 'Image_conv2')(I2)#32,3,relu
I4 = k.layers.Flatten(name = 'Image_flatten')(I3)

F1 = k.layers.Input(shape=(featurenumber,), name = 'Feature_Input')#Feature shape
F2 = k.layers.Dense(units = 16, activation = 'relu', name = 'Feature_dense1')(F1)#256,relu
F3 = k.layers.Dropout(rate = 0.1, name = 'Image_dropout1')(F2)#0.2
F4 = k.layers.Dense(units = 16, activation = 'relu', name = 'Feature_dense2')(F3)#256,relu
F5 = k.layers.Dropout(rate = 0.1, name = 'Image_dropout2')(F4)#0.2
F6 = k.layers.Dense(units = 16, activation = 'relu', name = 'Feature_dense3')(F5)#256,relu
F7 = k.layers.Dropout(rate = 0.1, name = 'Image_dropout3')(F6)#0.2

#O1 = k.layers.Dense(units = 10, name = 'Output_Input')(I4)
O1 = k.layers.concatenate([I4,F7])
O2 = k.layers.Dense(units = 16, activation = 'relu', name = 'Output_intermediate')(O1)#256,relu
O3 = k.layers.Dense(units = classnumber, activation = 'softmax', name = 'Output_classes')(O2)#numberofclasses, softmax

model = k.models.Model(inputs=[I1,F1], outputs=O3)

opt = k.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# checkpoints
checkpointer    = k.callbacks.ModelCheckpoint(filepath=outDir+'/bestweights.hdf5', monitor='val_loss', verbose=args.verbose, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy
# coitointerrotto = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=args.totEpochs, restore_best_weights=True)
logger          = k.callbacks.callbacks.CSVLogger(outDir+'epochs.log', separator=' ', append=False)
callbacks=[checkpointer, logger]
'''
### TRAIN ###

# train the neural network
start=time.time()
if args.aug:
	history = model.fit_generator(
		aug.flow(trainX, trainY, batch_size=args.bs), 
		validation_data=(testX, testY), 
		steps_per_epoch=len(trainX)//args.bs,	
		epochs=args.totEpochs, 
		callbacks=callbacks,
		initial_epoch = 0,
		verbose=args.verbose)
else:
'''
history = model.fit(
	[x1,x2], y, batch_size=args.bs, 
	#validation_data=(testX, testY), 
	epochs=epochs, 
	callbacks=callbacks,
	initial_epoch = 0,
	verbose=args.verbose,
	validation_split=args.testSplit)
#trainingTime=time.time()-start
#print('Training took',trainingTime/60,'minutes')
