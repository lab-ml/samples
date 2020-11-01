.PHONY: help
.DEFAULT_GOAL := help

docs: ## Render annotated HTML
	python ../../pylit/pylit.py --remove_empty_sections -s ../../pylit/pylit_docs.css -t ../../pylit/template_samples.html -d html -w samples

pages: ## Copy to lab-ml site
	@cd ../lab-ml.github.io; git pull
	cp -r html/* ../lab-ml.github.io/


help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
