
#Script that runs on the arguments -path and -epochnumber 

#to save a picture of val_loss and val_accuracy in comparison
#as a function of the changed imagedataset-size

#the input data has to be in form of folders inside the path directory where the epoch log files are placed in

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os, argparse
import seaborn as sns



parser = argparse.ArgumentParser(description='analyze the epoch history of multiple training results graphically')
parser.add_argument('-path', default='./binary/', help="The datapath in which there are outputdirectories from the training runs.")
parser.add_argument('-epochnumber', type=int, default=100, help="number of epochs")
args=parser.parse_args()


methodpath = args.path
epochnumber = args.epochnumber


a = os.listdir(methodpath)[0].find('on_')
if a != -1:
	o = a+3
	p = a+7
else:
	o=0
	p=-1
	print('there is no on_ in the output directory names')


loss_max = 3

columns = np.array([])
val_loss_list = np.array([])
loss_list = np.array([])
val_accuracy_list = np.array([])
accuracy_list = np.array([])



keys = []

for i in os.listdir(methodpath):
    b = i.find('binary_')
    c = i.find('_limit')
    if b != -1 and c != -1:
        key = i[b+7:c]
        if not key in keys:
            keys.append(key)
            
loss=pd.DataFrame(columns=keys, index=np.arange(3000),dtype = np.float32)

accuracy=pd.DataFrame(columns=keys,index=np.arange(3000),dtype = np.float32)

for i in os.listdir(methodpath):
    columns = np.append(columns, i+'val_loss')
    columns = np.append(columns, i+'loss')
    columns = np.append(columns, i+'val_accuracy')
    columns = np.append(columns, i+'accuracy')
    
    val_loss_list = np.append(val_loss_list,i+'val_loss')
    loss_list = np.append(loss_list,i+'loss')
    val_accuracy_list = np.append(val_accuracy_list,i+'val_accuracy')
    accuracy_list = np.append(accuracy_list,i+'accuracy')
    
index = np.arange(epochnumber)

df = pd.DataFrame(columns=columns, index = index, dtype = np.float32)

for i in os.listdir(methodpath):
    for x in os.listdir(methodpath+i):
        if x == 'epochs.log':
            b = np.array(pd.read_csv(methodpath+i+'/'+x, sep = ',', header=0),dtype=np.float32)
            d = i.find('binary_')
            f = i.find('_limit')
            key = i[d+7:f]
            limit = np.int(i[f+10:f+14])
            loss[key][limit] = b[:,1].min(axis=0)
            accuracy[key][limit] = b[:,2].max(axis=0)

            
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex = True)
fig.suptitle('best validation stats for the binary classifier', fontsize = 20)
axes[0].set_ylabel('loss',fontsize=14)
axes[1].set_ylabel('accuracy',fontsize=14)
axes[1].set_xlabel('Imageset size', fontsize=14)
#axes[1].set_xlim(0,2000)
for i in keys:
    axes[0].scatter(loss.index,loss[i], label=i)
    axes[1].scatter(accuracy.index,accuracy[i], label=i)

    
fig.legend(labels = keys, loc = 'lower center', ncol=3, labelspacing=0.,fontsize=14)
fig.savefig(methodpath[2:-1])
