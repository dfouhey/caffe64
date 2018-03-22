main:
	gcc -o caffe64 caffe64.s -nostdlib 
	strip caffe64

data:
	cd mnist && python dl.py

clsTrain:
	mkdir -p netWeights
	./caffe64 train mnist/network5.txt netWeights/weights.bin mnist/optimsetting.txt 60000 mnist/mnistX.txt mnist/mnistYn.txt

clsTest:
	./caffe64 test mnist/network5.txt netWeights/weights.bin 10000 mnist/mnistXe.txt mnist/p.txt

regTrain:
	mkdir -p netWeights
	./caffe64 train mnist/networkReg.txt netWeights/weightsreg.bin mnist/optimsetting.txt 60000 mnist/mnistX.txt mnist/mnistYn.txt

regTest:
	./caffe64 test mnist/networkReg.txt netWeights/weightsreg.bin 10000 mnist/mnistXe.txt mnist/preg.txt
