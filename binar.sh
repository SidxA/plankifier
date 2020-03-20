##!/bin/bash

for i in {64..256..64}
	do
	python3 binary.py -limit=100 -totEpochs=10 -number1=$i -number2=$i -number3=$i opt='rmsprop'
	done
