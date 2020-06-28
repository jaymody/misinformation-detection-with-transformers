.PHONY: clean lint style test

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.ipynb_checkpoints' -type d -exec rm -rf {} +

lint:
	pylint -E pipeline.py valerie scripts

style:
	black pipeline.py valerie tests scripts

test:
	rm -rf logs/tests
	mkdir -p logs/tests
	bash run_tests 2>&1 | tee logs/tests/test.log
