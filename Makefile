.PHONY: clean lint style test fetch fetch_zips unzip_zips fetch_external push_data push_zips push_external

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

fetch_zips:
	gsutil cp gs://valerie-bucket/data/phase1/train.zip data/phase1/train.zip
	# gsutil cp gs://valerie-bucket/data/phase2-1/train.zip data/phase2-1/train.zip
	gsutil cp gs://valerie-bucket/data/phase2-3/train.zip data/phase2-3/train.zip

push_zips:
	gsutil cp data/phase1/train.zip gs://valerie-bucket/data/phase1/train.zip
	# gsutil cp data/phase2-1/train.zip gs://valerie-bucket/data/phase2-1/train.zip
	gsutil cp data/phase2-3/train.zip gs://valerie-bucket/data/phase2-3/train.zip

unzip_zips:
	rm -rf data/phase1/raw
	mkdir -p data/phase1/raw data/phase1/raw
	unzip data/phase1/train.zip -d data/phase1/raw

	# rm -rf data/phase2-1/raw
	# mkdir -p data/phase2-1/raw data/phase2-1/raw
	# unzip data/phase2-1/train.zip -d data/phase2-1/raw

	rm -rf data/phase2-3/raw
	mkdir -p data/phase2-3/raw data/phase2-3/raw
	unzip data/phase2-3/train.zip -d data/phase2-3/raw

fetch:
	# phase2 trial
	mkdir -p data/phase2-trial
	gsutil cp -r gs://valerie-bucket/data/phase2-trial/raw data/phase2-trial

	# phase2 validation
	mkdir -p data/phase2-validation
	gsutil cp -r gs://valerie-bucket/data/phase2-validation/raw data/phase2-validation

	# phase2 validation 10, 100, 500
	gsutil cp -r gs://valerie-bucket/data/phase2-validation/10 data/phase2-validation
	gsutil cp -r gs://valerie-bucket/data/phase2-validation/100 data/phase2-validation
	gsutil cp -r gs://valerie-bucket/data/phase2-validation/500 data/phase2-validation

	# phase1
	gsutil cp gs://valerie-bucket/data/phase1/raw/metadata.json data/phase1/raw/metadata.json

	# phase2
	# gsutil cp gs://valerie-bucket/data/phase2-1/raw/metadata.json data/phase2-1/raw/metadata.json
	gsutil cp gs://valerie-bucket/data/phase2-3/raw/metadata.json data/phase2-3/raw/metadata.json

push:
	# phase2 trial
	gsutil cp -r data/phase2-trial/raw gs://valerie-bucket/data/phase2-trial/raw

	# phase2 validation
	gsutil cp -r data/phase2-validation/raw gs://valerie-bucket/data/phase2-validation/raw

	# phase2 validation 10, 100, 500
	gsutil cp -r data/phase2-validation/10 gs://valerie-bucket/data/phase2-validation/10
	gsutil cp -r data/phase2-validation/100 gs://valerie-bucket/data/phase2-validation/100
	gsutil cp -r data/phase2-validation/500 gs://valerie-bucket/data/phase2-validation/500

	# phase1
	gsutil cp data/phase1/raw/metadata.json gs://valerie-bucket/data/phase1/raw/metadata.json

	# phase2
	# gsutil cp data/phase2-1/raw/metadata.json gs://valerie-bucket/data/phase2-1/raw/metadata.json
	gsutil cp data/phase2-3/raw/metadata.json gs://valerie-bucket/data/phase2-3/raw/metadata.json

fetch_external:
	gsutil cp gs://valerie-bucket/data/external/2018-12-fake-news-top-50/data/top_2018.csv data/external/2018-12-fake-news-top-50/data/top_2018.csv
	gsutil cp gs://valerie-bucket/data/external/2018-12-fake-news-top-50/data/sites_2016.csv data/external/2018-12-fake-news-top-50/data/sites_2016.csv
	gsutil cp gs://valerie-bucket/data/external/2018-12-fake-news-top-50/data/sites_2017.csv data/external/2018-12-fake-news-top-50/data/sites_2017.csv
	gsutil cp gs://valerie-bucket/data/external/2018-12-fake-news-top-50/data/sites_2018.csv data/external/2018-12-fake-news-top-50/data/sites_2018.csv

	gsutil cp gs://valerie-bucket/data/external/fake-news/train.csv data/external/fake-news/train.csv

	gsutil cp gs://valerie-bucket/data/external/FakeNewsNet/dataset/politifact_fake.csv data/external/FakeNewsNet/dataset/politifact_fake.csv
	gsutil cp gs://valerie-bucket/data/external/FakeNewsNet/dataset/politifact_real.csv data/external/FakeNewsNet/dataset/politifact_real.csv
	gsutil cp gs://valerie-bucket/data/external/FakeNewsNet/dataset/gossipcop_fake.csv data/external/FakeNewsNet/dataset/gossipcop_fake.csv
	gsutil cp gs://valerie-bucket/data/external/FakeNewsNet/dataset/gossipcop_real.csv data/external/FakeNewsNet/dataset/gossipcop_real.csv

	gsutil cp gs://valerie-bucket/data/external/george-mcintire/fake_or_real_news.csv data/external/george-mcintire/fake_or_real_news.csv

	gsutil cp gs://valerie-bucket/data/external/ISOT/Fake.csv data/external/ISOT/Fake.csv
	gsutil cp gs://valerie-bucket/data/external/ISOT/True.csv data/external/ISOT/True.csv

	gsutil cp gs://valerie-bucket/data/external/liar/train.tsv data/external/liar/train.tsv
	gsutil cp gs://valerie-bucket/data/external/mrisdal/fake.csv data/external/mrisdal/fake.csv

push_external:
	gsutil cp data/external/2018-12-fake-news-top-50/data/top_2018.csv gs://valerie-bucket/data/external/2018-12-fake-news-top-50/data/top_2018.csv
	gsutil cp data/external/2018-12-fake-news-top-50/data/sites_2016.csv gs://valerie-bucket/data/external/2018-12-fake-news-top-50/data/sites_2016.csv
	gsutil cp data/external/2018-12-fake-news-top-50/data/sites_2017.csv gs://valerie-bucket/data/external/2018-12-fake-news-top-50/data/sites_2017.csv
	gsutil cp data/external/2018-12-fake-news-top-50/data/sites_2018.csv gs://valerie-bucket/data/external/2018-12-fake-news-top-50/data/sites_2018.csv

	gsutil cp data/external/fake-news/train.csv gs://valerie-bucket/data/external/fake-news/train.csv

	gsutil cp data/external/FakeNewsNet/dataset/politifact_fake.csv gs://valerie-bucket/data/external/FakeNewsNet/dataset/politifact_fake.csv
	gsutil cp data/external/FakeNewsNet/dataset/politifact_real.csv gs://valerie-bucket/data/external/FakeNewsNet/dataset/politifact_real.csv
	gsutil cp data/external/FakeNewsNet/dataset/gossipcop_fake.csv gs://valerie-bucket/data/external/FakeNewsNet/dataset/gossipcop_fake.csv
	gsutil cp data/external/FakeNewsNet/dataset/gossipcop_real.csv gs://valerie-bucket/data/external/FakeNewsNet/dataset/gossipcop_real.csv

	gsutil cp data/external/george-mcintire/fake_or_real_news.csv gs://valerie-bucket/data/external/george-mcintire/fake_or_real_news.csv

	gsutil cp data/external/ISOT/Fake.csv gs://valerie-bucket/data/external/ISOT/Fake.csv
	gsutil cp data/external/ISOT/True.csv gs://valerie-bucket/data/external/ISOT/True.csv

	gsutil cp data/external/liar/train.tsv gs://valerie-bucket/data/external/liar/train.tsv
	gsutil cp data/external/mrisdal/fake.csv gs://valerie-bucket/data/external/mrisdal/fake.csv
