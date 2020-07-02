.PHONY: clean lint style test fetch fetch_zips

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

fetch:
	gsutil cp -r gs://valerie-bucket/data/phase2-trial/raw data/phase2-trial/raw
	# python phase2_trial_data.py --trial_data_raw_dir data/phase2-trial/raw --output_dir data/phase2-trial

	gsutil cp gs://valerie-bucket/data/phase1/raw/metadata.json data/phase1/raw/metadata.json
	gsutil cp gs://valerie-bucket/data/phase2-1/raw/metadata.json data/phase2/raw/metadata.json

fetch_zips:
	gsutil cp gs://valerie-bucket/data/phase1/train.zip data/phase1/train.zip
	gsutil cp gs://valerie-bucket/data/phase2-1/train.zip data/phase2-1/train.zip

	# mkdir -p data/phase1/raw data/phase2/raw
	# unzip data/phase1/train.zip -d data/phase1/raw
	# unzip data/phase2/train.zip -d data/phase2/raw
