import os, sys, pathlib, glob, time, datetime, argparse
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import keras as k
from keras.layers import Activation, Dense
import matplotlib.pyplot as plt, seaborn as sns
from src import helper_models, helper_data
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

###try gpu
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser(description='Train a model on zooplankton images')
parser.add_argument('-datapath', default='./data/', help="Print many messages on screen.")
#parser.add_argument('-datakind', default='image', choices=['mixed','image','tsv'], help="If tsv, expect a single tsv file; if images, each class directory has only images inside; if mixed, expect a more complicated structure defined by the output of SPCConvert")
parser.add_argument('-outpath', default='./out/', help="Print many messages on screen.")
#parser.add_argument('-verbose', action='store_true', help="Print many messages on screen.")
#parser.add_argument('-plot', action='store_true', help="Plot loss and accuracy during training once the run is over.")
parser.add_argument('-totEpochs', type=int, default=5, help="Total number of epochs for the training")
parser.add_argument('-opt', default='sgd', help="Choice of the minimization algorithm (sgd,adam)")
parser.add_argument('-bs', type=int, default=32, help="Batch size")
parser.add_argument('-lr', type=float, default=0.0001, help="Learning Rate")
#parser.add_argument('-height', type=int, default=128, help="Image height")
#parser.add_argument('-width', type=int, default=128, help="Image width")
#parser.add_argument('-depth', type=int, default=3, help="Number of channels")
#parser.add_argument('-testSplit', type=float, default=0.2, help="Fraction of examples in the validation set")
parser.add_argument('-aug', default = True, help="Perform data augmentation.")
#parser.add_argument('-resize', choices=['keep_proportions','acazzo'], default='keep_proportions', help='The way images are resized')
#parser.add_argument('-model', choices=['mlp','conv2','smallvgg'], default='conv2', help='The model. MLP gives decent results, conv2 is the best, smallvgg overfits (*validation* accuracy oscillates).')
#parser.add_argument('-layers',nargs=2, type=int, default=[256,128], help="Layers for MLP")
#parser.add_argument('-load', default=None, help='Path to a previously trained model that should be loaded.')
#parser.add_argument('-override_lr', action='store_true', help='If true, when loading a previously trained model it discards its LR in favor of args.lr')
#parser.add_argument('-initial_epoch', type=int, default=0, help='Initial epoch of the training')
parser.add_argument('-limit', type = int, default=0, help='number of images')
parser.add_argument('-key', type = str, default='dinobryon', help='To be identified class.')
parser.add_argument('-number1',type = int, default = 256, help='nodenumbers of the first cnn layers')
parser.add_argument('-number2',type = int, default = 128, help='nodenumbers of the second cnn layers')
parser.add_argument('-number3',type = int, default = 64, help='nodenumbers of the third cnn layers')
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

print('\nRunning',sys.argv[0],sys.argv[1:])

# Create a unique output directory
now = datetime.datetime.now()
dt_string = now.strftime("%Y-%m-%d_%Hh%Mm%Ss")
filename = 'binary_'+'limit:'+np.str(args.limit)+'_'+dt_string
outDir = args.outpath+'/'+filename+'/'
pathlib.Path(outDir).mkdir(parents=True, exist_ok=True)
fsummary=open(outDir+'args.txt','w')
print(args, file=fsummary); fsummary.flush()


########
# DATA #
########

imagesize = 128
datapath = args.datapath
learning_rate = 0.0001
batchsize=8
epochs=args.totEpochs
key = args.key
limit = args.limit
depth = 3
lr=.0001

def datainput(dapath,key,limit,imagesize):
	
	# classes as description of dataintake
	classes = {'name': [ name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name)) ]}
	df = pd.DataFrame(columns=['label', 'npimage'])

	#numbers of examples of the selected to be classified class
	keynumber = (len([name for name in os.listdir(datapath+key+'/training_data/')]))
	keypic = np.arange(keynumber)
	
	#if you would want only a random portion of the key classes images use the limit argument
	if limit != 0 and limit <= keynumber:
		rng = np.random.default_rng()
		rand = rng.choice(keynumber, size=limit, replace=False)
		keypic = [keypic[x] for x in rand]
		keynumber = limit
	elif limit != 0 and limit >= keynumber:
		raise NotImplementedError('There are not enough images as the selected Limit')

	#take in of the selected amount of key class images
	classpath = datapath + key +'/training_data/'
	classImages = os.listdir(classpath)
	classImages = [classImages[x] for x in keypic]

	for imagename in classImages:
			imagePath = datapath+key+'/training_data/'+imagename
			image = Image.open(imagePath)
			image,rescaled = helper_data.ResizeWithProportions(image, imagesize)
			npimage = np.array(image.copy() , dtype=np.float32)/255.0
			df=df.append({'label':1, 'npimage':npimage}, ignore_index=True)
			image.close()
				

	#there are going to be roughly 50/50 key/notkey,
	#first all notkey classes with not enough images are beeing taken in
	#and the rest is filled up balanced
	#the remaining classes beeing made to the balanced notkey class
	classes['name'].remove(key)
	classnumber = int(keynumber / len(classes['name']))
	remaining = keynumber
	remaining_classes = len(classes['name'])

	#sort classes by numbers of examples
	for i in classes['name']:
		classpath = datapath + i +'/training_data/'
		if len(os.listdir(classpath)) >= classnumber:
			classes['name'].remove(i)
			classes['name'].append(i)

	#first take in all with not enough examples then fill up with the rest
	for i in classes['name']:
		classpath = datapath + i +'/training_data/'
		classImages = os.listdir(classpath)
		number = len(os.listdir(classpath))
		
		if number >= classnumber:
			count = int(remaining/remaining_classes)
			if number <= count: count = number 
			rng = np.random.default_rng()
			rand = rng.choice(number, size=count, replace=False)
			classImages = [classImages[x] for x in rand]
			
		for imagename in classImages:
				imagePath = datapath+i+'/training_data/'+imagename
				image = Image.open(imagePath)
				image,rescaled = helper_data.ResizeWithProportions(image, imagesize)
				npimage = np.array(image.copy() , dtype=np.float32)/255.0
				df=df.append({'label':0, 'npimage':npimage}, ignore_index=True)
				image.close()
		
		remaining -= len(classImages)  
		remaining_classes -= 1
	outstring = 'Datainput took '+np.str(keynumber)+' '+key+' and a balanced mix of '+', '.join(classes['name'])
	print(outstring)
	return np.stack(df['npimage']), np.array(np.stack(df['label']),dtype = bool)
	
data, labels = datainput(datapath,key,limit,imagesize)

if K.image_data_format() == 'channels_first':
    input_shape = (3, imagesize, imagesize)
else:
    input_shape = (imagesize,imagesize, 3)

(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=0.2, random_state=42)


model = Sequential()
model.add(k.layers.Conv2D(args.number1, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(k.layers.Conv2D(args.number2, (3, 3)))
model.add(Activation('relu'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(k.layers.Conv2D(args.number3, (3, 3)))
model.add(Activation('relu'))
model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(k.layers.Flatten())
model.add(k.layers.Dense(64))
model.add(Activation('relu'))
model.add(k.layers.Dropout(0.5))
model.add(k.layers.Dense(1))
model.add(Activation('sigmoid'))

              
'''              
model = helper_models.Conv2Layer.build(width=imagesize, height=imagesize, depth=depth, classes = 2)
model.add(Dense(1, activation='sigmoid'))
opt = k.optimizers.SGD(lr=lr, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
'''

#DATA augmentation

aug = ImageDataGenerator(
	rotation_range=360,     width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.3,
	zoom_range=0.2, 		horizontal_flip=True,
	vertical_flip=True,		
	)


if args.opt=='adam':
	opt = keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, amsgrad=True)
elif args.opt=='sgd':
	opt = keras.optimizers.SGD(lr=args.lr, nesterov=True)
elif args.opt=='rmsprop':
	opt = keras.optimizers.RMSprop(lr=args.lr, rho = 0.9)
		
		
		
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
if args.aug:
	history = model.fit_generator(
		aug.flow(trainX, trainY, batch_size=args.bs), 
		validation_data=(testX, testY), 
		epochs=args.totEpochs, 
		verbose = 1)
else:
	history = model.fit(
		trainX, trainY, batch_size=args.bs, 
		validation_data=(testX, testY), 
		epochs=args.totEpochs, 
		verbose=1)

model.save_weights(outDir+"model.h5")

output = pd.DataFrame(history.history)
hist_csv_file = outDir+'epochs.log'
with open(hist_csv_file, mode='w') as f:
    output.to_csv(f)

'''
# checkpoints
checkpointer    = keras.callbacks.ModelCheckpoint(filepath=outDir+'/bestweights.hdf5', monitor='val_loss', verbose=0, save_best_only=True) # save the model at every epoch in which there is an improvement in test accuracy
# coitointerrotto = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', patience=args.totEpochs, restore_best_weights=True)
logger          = keras.callbacks.callbacks.CSVLogger(outDir+'epochs.log', separator=' ', append=False)
callbacks=[checkpointer, logger]

### evaluate the network
print("[INFO] evaluating network...")
if args.aug:
	predictions = model.predict(testX)
else:
	predictions = model.predict(testX, batch_size=args.bs)
clrep=classification_report(testY.argmax(axis=0), predictions.argmax(axis=1))
print(clrep)



# Identify the easiest prediction and the worse mistake
i_maxconf_right=-1; i_maxconf_wrong=-1
maxconf_right  = 0; maxconf_wrong  = 0
for i in range(test_size):
	# Spot easiestly classified image (largest confidence, and correct classification)
	if testY.argmax(axis=1)[i]==predictions.argmax(axis=1)[i]: # correct classification
		if predictions[i][predictions[i].argmax()]>maxconf_right: # if the confidence on this prediction is larger than the largest seen until now
			i_maxconf_right = i
			maxconf_right   = predictions[i][predictions[i].argmax()]
	# Spot biggest mistake (largest confidence, and incorrect classification)
	else: # wrong classification
		if predictions[i][predictions[i].argmax()]>maxconf_wrong:
			i_maxconf_wrong=i
			maxconf_wrong=predictions[i][predictions[i].argmax()]


# Confidences of right and wrong predictions
confidences = predictions.max(axis=1) # confidence of each prediction made by the classifier
whether = np.array([1 if testY.argmax(axis=1)[i]==predictions.argmax(axis=1)[i] else 0 for i in range(len(predictions))]) #0 if wrong, 1 if right
confidences_right = confidences[np.where(testY.argmax(axis=1)==predictions.argmax(axis=1))[0]]
confidences_wrong = confidences[np.where(testY.argmax(axis=1)!=predictions.argmax(axis=1))[0]]


# Abstention accuracy
thresholds = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.997,0.999,0.9995,0.9999,0.99995,0.99999], dtype=np.float)
accs,nconfident = np.ndarray(len(thresholds), dtype=np.float), np.ndarray(len(thresholds), dtype=np.int)
for i,thres in enumerate(thresholds):
	confident     = np.where(confidences>thres)[0]
	nconfident[i] = len(confident)
	accs      [i] = whether[confident].sum()/nconfident[i] if nconfident[i]>0 else np.nan


##########
# OUTPUT #
##########


# Save classification report
with open(outDir+'/classification_report.txt','w') as frep:
	print(clrep, file=frep)
	# For each class, write down what it was confused with
	print('\nLet us see with which other taxa each class gets confused.', file=frep)
	for ic,c in enumerate(classes['name']):
		print("{:18}: ".format( classes['name'][ic]), end=' ', file=frep)
		ic_examples = np.where(testY.argmax(axis=1)==ic)[0] # examples in the test set with label ic
		ic_predictions = predictions[ic_examples].argmax(axis=1)
		histo = np.histogram(ic_predictions, bins=np.arange(classes['num']+1))[0]/len(ic_examples)
		ranks = np.argsort(histo)[::-1]
		# ic_classes = [classes['name'][ranks[i]] for i in range(classes['num'])]
		for m in range(5): # Print only first few mistaken classes
			print("{:18}({:.2f})".format( classes['name'][ranks[m]],histo[ranks[m]]), end=', ', file=frep)
		print('...', file=frep)

# Table with abstention data
print('threshold accuracy nconfident', file=open(outDir+'/abstention.txt','w'))
fabst=open(outDir+'/abstention.txt','a')
for i in range(len(thresholds)):
	print('{}\t{}\t{}'.format(thresholds[i],accs[i],nconfident[i]), file=fabst)
fabst.close()


### IMAGES ###

#outputfiles of the augmented pictures

for i in range(0,9):
	npimage = testX[i]
	npimage.reshape((args.width,args.height,args.depth))	
	npimage=np.rint(npimage*256).astype(np.uint8)
	image=Image.fromarray(npimage)
	plt.subplot(330 + 1 + i)
	plt.imshow(image, cmap=plt.get_cmap('gray'))
	plt.savefig(outDir+'/original.png')

for X_batch, y_batch in aug.flow(testX, testY, batch_size=9):
	# Show 9 images
	for i in range(0,9):
		plt.subplot(330 + 1 + i)
		plt.imshow(X_batch[i].reshape(args.width,args.height,args.depth))

	plt.savefig(outDir+'/augmented.png')
	break
    
def plot_npimage(npimage, ifig=0, width=64, height=64, depth=3, title='Yet another image', filename=None):
	plt.figure(ifig)
	npimage.reshape((args.width,args.height,args.depth))	
	npimage=np.rint(npimage*256).astype(np.uint8)
	image=Image.fromarray(npimage)
	plt.title(title)
	plt.imshow(image)
	if filename!=None:
		plt.savefig(filename)



# Image of the easiest prediction
plot_npimage(testX[i_maxconf_right], 0, args.width, args.height, args.depth, 
	title='Prediction: {}, Truth: {}\nConfidence:{:.2f}'.format(classes['name'][ predictions[i_maxconf_right].argmax() ], 
																classes['name'][ testY      [i_maxconf_right].argmax() ],
																confidences[i_maxconf_right]),
	filename=outDir+'/easiest-prediction.png')

# Image of the worst prediction
plot_npimage(testX[i_maxconf_wrong], 1, args.width, args.height, args.depth, 
	title='Prediction: {}, Truth: {}\nConfidence:{:.2f}'.format(classes['name'][ predictions[i_maxconf_wrong].argmax() ], 
																classes['name'][ testY      [i_maxconf_wrong].argmax() ],
																confidences[i_maxconf_wrong]),
	filename=outDir+'/worst-prediction.png')


# Plot loss during training
plt.figure(2)
plt.title('Model loss during training')
simulated_epochs=len(history.history['loss']) #If we did early stopping it is less than args.totEpochs
plt.plot(np.arange(1,simulated_epochs+1),history.history['loss'], label='train')
plt.plot(np.arange(1,simulated_epochs+1),history.history['val_loss'], label='test')
plt.xlabel('epoch')
plt.xlabel('loss')
plt.legend()
plt.savefig(outDir+'/loss.png')

# Plot accuracy during training
plt.figure(3)
plt.title('Model accuracy during training')
plt.ylim((0,1))
plt.plot(np.arange(1,simulated_epochs+1),history.history['accuracy'], label='train')
plt.plot(np.arange(1,simulated_epochs+1),history.history['val_accuracy'], label='test')
plt.plot(np.arange(1,simulated_epochs+1),np.ones(simulated_epochs)/classes['num'], label='random', color='black', linestyle='-.')
plt.xlabel('epoch')
plt.xlabel('loss')
plt.grid(axis='y')
plt.legend()
plt.savefig(outDir+'/accuracy.png')

# Scatter plot and density of correct and incorrect predictions (useful for active and semi-supervised learning)
plt.figure(4)
plt.title('Correct/incorrect predictions and their confidence')
sns.distplot(confidences_right, bins=20, label='Density of correct predictions', color='green')
sns.distplot(confidences_wrong, bins=20, label='Density of wrong   predictions', color='red')
plt.plot(confidences, whether, 'o', label='data (correct:1, wrong:0)', color='black', markersize=1)
plt.xlabel('confidence')
plt.xlim((0,1))
plt.ylim(bottom=-0.2)
plt.legend()
plt.savefig(outDir+'/confidence.png')


# Plot Abstention
plt.figure(5)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=.7)
ax1=plt.subplot(2, 1, 1)
ax1.set_ylim((0,1))
plt.title('Abstention')
plt.ylabel('Accuracy after abstention')
plt.xlabel('Threshold')
plt.plot(thresholds, accs, color='darkred')
plt.grid(axis='y')
ax2=plt.subplot(2, 1, 2)
ax2.set_ylim((0.1,test_size*1.5))
ax2.set_yscale('log')
plt.ylabel('Remaining data after abstention')
plt.xlabel('Threshold')
plt.plot(thresholds, nconfident, color='darkred')
plt.grid(axis='y')
plt.savefig(outDir+'/abstention.png')


'''
