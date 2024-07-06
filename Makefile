
.PHONY: clean-pyc clean-build

help:
	@echo "clean-pyc -- remove auxiliary python files"
	@echo "clean -- total cleaning of project files"
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean: clean-pyc
