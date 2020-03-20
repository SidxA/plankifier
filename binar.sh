##!/bin/bash

for i in {64..256..64}
	do
	python3 binary.py -limit=100 -totEpochs=10 -opt='rmsprop' -number1=$i -number2=$i -number3=$i 
	done
