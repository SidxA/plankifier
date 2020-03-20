##!/bin/bash

python3 binary.py -limit=0 -totEpochs=3000 -opt='rmsprop' -number1=256 -number2=256 -number3=128 -aug=True

for i in {50..3000..50}
	do
	python3 binary.py -limit=$i -totEpochs=300 -opt='rmsprop' -number1=256 -number2=256 -number3=128 -aug=True
	done
