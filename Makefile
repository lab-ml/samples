.PHONY: help
.DEFAULT_GOAL := help

cifr10: ## Run cifar10
	python -m pytorch.cifr10.cifar10

rnn: ## RNN
	python pytorch/rnn/rnn.py

gan: ## GAN
	python pytorch/gan/simple_gan.py

mnist_configs: ## MNIST Configs
	python pytorch/mnist/configs.py

mnist_hyperparam_tuning: ## MNIST Hyper-parameter tuning
	python pytorch/mnist/hyperparameter_tunining.py

mnist_indexed_logs: ## MNIST Indexed logs
	python pytorch/mnist/indexed_logs.py

mnist_latest: ## MNIST
	python pytorch/mnist/lab_latest.py

mnist_v1: ## MNIST v1
	python pytorch/mnist/lab_v1.py


mnist: mnist_configs mnist_hyperparam_tuning mnist_indexed_logs mnist_v1 mnist_latest ## All MNIST

pytorch: cifr10 rnn gan mnist ## All PyTorch

sklearn: ## SKLearn sample
	python scikitlearn/scikit-learn.py

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'


all: pytorch  sklearn
