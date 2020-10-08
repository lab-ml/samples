.PHONY: help
.DEFAULT_GOAL := help

cifar10: ## Run cifar10
	python -m samples.pytorch.cifr10.cifar10

rnn: ## RNN
	python samples/pytorch/rnn/rnn.py

gan: ## GAN
	python samples/pytorch/gan/simple_gan.py

mnist_latest: ## MNIST Configs
	python samples/pytorch/mnist/configs.py

mnist_v1: ## MNIST v1
	python samples/pytorch/mnist/lab_v1.py

mnist: mnist_v1 mnist_latest ## All MNIST

pytorch: cifar10 rnn gan mnist ## All PyTorch

sklearn: ## SKLearn sample
	python samples/scikitlearn/scikit-learn.py

docs: ## Render annotated HTML
	python ../../pylit/pylit.py --remove_empty_sections -s ../../pylit/pylit_docs.css -t ../../pylit/template_docs.html -d html -w samples

pages: ## Copy to lab-ml site
	@cd ../lab-ml.github.io; git pull
	cp -r html/* ../lab-ml.github.io/


help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'


all: pytorch  sklearn
