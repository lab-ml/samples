cifr10:
	python -m pytorch.cifr10.cifar10

rnn:
	python pytorch/rnn/rnn.py

gan:
	python pytorch/gan/simple_gan.py

mnist_configs:
	python pytorch/mnist/configs.py

mnist_hyperparam_tuning:
	python pytorch/mnist/hyperparameter_tunining.py

mnist_indexed_logs:
	python pytorch/mnist/indexed_logs.py

mnist_latest:
	python pytorch/mnist/lab_latest.py

mnist_v1:
	python pytorch/mnist/lab_v1.py


mnist: mnist_configs mnist_hyperparam_tuning mnist_indexed_logs mnist_v1 mnist_latest

pytorch: cifr10 rnn gan mnist

sklearn:
	python scikitlearn/scikit-learn.py


all: pytorch  sklearn