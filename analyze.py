import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os, argparse
import seaborn as sns


#Script that runs on the arguments -path and -epochnumber 

#to save a picture contaning the logarithmic evolution of loss, accuracy, val_loss and val_accuracy
#aswell as the minimal val_loss as a function of the changed parameter

#the input data has to be in form of folders inside the path directory where the epoch log files are placed in

parser = argparse.ArgumentParser(description='analyze the epoch history of multiple training results graphically')
parser.add_argument('-path', default='./rescale/', help="The datapath in which there are outputdirectories from the training runs.")
parser.add_argument('-epochnumber', type=int, default=100, help="number of epochs")
args=parser.parse_args()


methodpath = args.path
epochnumber = args.epochnumber

a = os.listdir(methodpath)[0].find('on_')
if a != -1:
	o = a+3
	p = a+8
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
            a = pd.read_csv(methodpath+i+'/'+x, sep = ' ')
            df[i+'val_loss'] = a['val_loss']
            df[i+'loss'] = a['loss']
            df[i+'val_accuracy'] = a['val_accuracy']
            df[i+'accuracy'] = a['accuracy']

val_loss_list = np.sort(val_loss_list)
loss_list = np.sort(loss_list)
val_accuracy_list = np.sort(val_accuracy_list)
accuracy_list = np.sort(accuracy_list)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
fig.suptitle(methodpath[2:-1]+' logarithmic epoch evolution', fontsize = 24)
cmap = plt.cm.Spectral

axes[1, 0].set_ylim(1/len(os.listdir(methodpath)),1)
axes[1, 1].set_ylim(1/len(os.listdir(methodpath)),1)
axes[0, 0].set_ylim(0,loss_max)
axes[0, 1].set_ylim(0,loss_max)


df[val_loss_list].plot(cmap = cmap, ax = axes[0,0], title='val_loss',logx = True)
#axes[0,0].legend([i[o:p] for i in val_loss_list])

df[loss_list].plot(cmap = cmap, ax = axes[0,1], title='loss',logx = True)
#axes[0,1].legend([i[o:p] for i in loss_list])

df[val_accuracy_list].plot(cmap = cmap, ax = axes[1,0], title='val_accuracy',logx = True)
#axes[1,0].legend([i[o:p] for i in val_accuracy_list])

df[accuracy_list].plot(cmap = cmap, ax = axes[1,1], title='accuracy',logx = True)
#axes[1,1].legend([i[o:p] for i in accuracy_list])

axes[0,0].legend().set_visible(False)
axes[0,1].legend().set_visible(False)
axes[1,0].legend().set_visible(False)
axes[1,1].legend().set_visible(False)

#lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
#lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
fig.legend(labels = [i[o:p] for i in val_loss_list], loc = 'lower center', ncol=3, labelspacing=0. )
fig.savefig(methodpath[2:-1])



fig, ax = plt.subplots(figsize = (16,9))
fig.suptitle(methodpath[2:-1]+' val_loss per changed parameter', fontsize = 24)
plt.xticks(rotation=45)
df2 = pd.DataFrame(index= ['name','max','min','last'])
for i in df[val_loss_list]:
    name = i[o:p]
    maxi = 0
    mini=5
    last = 0
    for e in df[val_loss_list][i]:
        if e <= mini:
            mini = e
        if e >= maxi:
            maxi = e
    last = (df[val_loss_list][i].iloc[-1])
    df2[name] = [name,maxi,mini,last]

#ax.bar(x = df2.loc['name'], bottom = df2.loc['min'], height = df2.loc['max'])
#ax.bar(x = df2.loc['name'], bottom = df2.loc['last'], height = 0.01)
#ax.bar(x = df2.loc['name'], bottom = df2.loc['min'], height = 0.01)
ax.scatter(x = df2.loc['name'], y = df2.loc['min'])
#ax.set_ylim(0,3)
ax.set_yscale('log')
ax.set_xlabel('Augmentation parameter', fontsize = 18)
ax.set_ylabel('val_loss', fontsize = 18)


loss = 10
number = 0
z = 0
for i in df2.loc['min']:
    if i <= loss:
        loss = i
        number =z
    z +=1
string = 'Val_Loss: Minimum of '+np.str(np.around(loss,3))+' at '+np.str(df2.loc['name'][number])
ax.legend([string])

ax.legend([string])
fig.savefig(methodpath[2:-1]+'_val_loss')
