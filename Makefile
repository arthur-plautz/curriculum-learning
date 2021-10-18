update_requirements:
	pip freeze | sed "s/==.*//g" > requirements.txt
