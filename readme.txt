Zooplankton classification software kit

---------------------------------------------------------------
	contents
	
		name		run						description

0		convnet		python3 convnet.py		script that builds a cnn and trains it on a given dataset of images
1		features	python3 features.py		script that builds a model consisting of a cnn for images and a mlp for tabular feture data
2		binary		python3 binary.py		script that build a cnn for binary classification for a chosen class and mixes a balance of other classes
3		analyze		python3 analyze.py		script that reads in training output and visualizes logarithmic time evolution and the val_loss differences between hyperparameters
----------------------------------------------------------------

0: convnet

the most important parsed parameter is datapath, all other default values should lead to decent results
if one argument is changed from the default value, the output name will contain the change

example for a training run with 100 epochs, the adam optimizer with amsgrad, a specific data directory and live training output into a log file in the dir the script runs in:
python3 convnet.py -datapath='~/specific/data/' -totEpochs=100 -opt='adam_2' -verbose=1 >> trainingresults.log &

	argument		type		default			description
	
	cpu				bool		False			performs training only on cpus
	gpu				bool		False			performs training on gpus
	datapath		str			'./data/'		directory which must contain classes as subdirectories with a directory 'training_images' inside
	outpath			str			'./out/'		(created) directory for the training output, a subdirectory will be created with the parameters of the run inside the name
	verbose			int			1				one of [0,1,2] for amount of output of training documentation
	totEpochs		int			10				total number of epochs for the training
	opt				str			'sgd_1'			Choice of the minimization algorithm	
	bs				int			8				Batch size
	lr				float		0.0001			Learning Rate
	height			int			128				Image height, must be the same as width
	width			int			128				Image width, must be the same as height
	depth			int			3				Number of channels (3 for RGB)
	testSplit		float		0.2				Fraction of examples in the validation set
	aug				bool		True			Perform data augmentation
	augtype			string		'standard'		Augmentation type
	augparameter	float		0				Augmentation parameter when testing one type of augmentaion, ignored for standard augmentation

implemented optimizer choices (the learning rate is set for all by -lr):

	-opt			description
	
	'adam_1'		Adam without amsgrad, beta_1=0.9, beta2=0.999
	'adam_2'		Adam with amsgrad, beta_1=0.9, beta2=0.999
	'sgd_1'			stochastic gradient descent without nesterov
	'sgd_2'			stochastic gradient descent with nesterov
	'sgd_3'			stochastic gradient descent with nesterov and momentum of 0.1
	'sgd_4'			stochastic gradient descent without nesterov and momentum of 0.1
	'rmsprop'		RMSprop with rho = 0.9
	'adagrad'		Adagrad
	'adadelta'		Adadelta with rho = 0.95
	'adamax'		Adamax with beta_1 = 0.9, beta_2 = 0.999
	'nadam'			Nadam with  beta_1 = 0.9, beta_2 = 0.999

implemented choices for individual data augmentation:

	-augtype		-augparameter description
	
	'rotate'		Degree range for random rotations
	'v_shift'		width shift: fraction of total width, if < 1, or pixels if >= 1
	'h_shift'		height shift: fraction of total height, if < 1, or pixels if >= 1
	'shear'			Shear Intensity (Shear angle in counter-clockwise direction in degrees)
	'zoom'			Range for random zoom [lower, upper] = [1-args.augparameter, 1+args.augparameter]
	'h_flip'		enables flippling, no -augparameter required
	'v_flip'		enables flippling, no -augparameter required
	'brightness'	Range for picking a brightness shift value from [lower, upper] = [args.augparameter,1-args.augparameter]
	'rescale'		multiply the data by the value provided after applying all other transformations
	
	'standard'		performs mixed augmentation with rotation_range=360, width_shift_range=0.2,
					height_shift_range=0.2, shear_range=0.3, zoom_range=0.2, horizontal_flip=True, vertical_flip=True
					no -augparameter required
					
----------------------------------------------------------------

1: features

this script not only takes the image data as an input but also the tabular features files, which have to be in the class directories

the most important parsed parameter is datapath, all other default values should lead to decent results


	argument		type		default			description
	
	cpu				bool		False			performs training only on cpus
	gpu				bool		False			performs training on gpus
	datapath		str			'./small_data/'	directory which must contain classes as subdirectories with a directory 'training_images' inside
	outpath			str			'./out/'		(created) directory for the training output, a subdirectory will be created with the parameters of the run inside the name
	verbose			int			1				one of [0,1,2] for amount of output of training documentation
	totEpochs		int			10				total number of epochs for the training
	bs				int			8				Batch size
	lr				float		0.0001			Learning Rate
	height			int			128				Image height, must be the same as width
	width			int			128				Image width, must be the same as height
	depth			int			3				Number of channels (3 for RGB)
	testSplit		float		0.2				Fraction of examples in the validation set

so far, SGD is implemented and no data augmentation is performed
