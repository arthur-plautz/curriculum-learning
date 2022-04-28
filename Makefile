update_requirements:
	pip freeze | sed "s/==.*//g" > requirements.txt

install:
	python setup.py install