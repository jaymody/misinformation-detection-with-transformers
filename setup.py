import setuptools

with open('README.md', 'r') as fi:
    long_description = fi.read()

setuptools.setup(
	name='valerie',
	version='0.1',
	author='Jay Mody',
	author_email='jaykmody@gmail.com',
	description='Fake news detection.',
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	url='https://github.com/jaymody/valerie',
	packages=setuptools.find_packages(),
	install_requires=[],
	classifiers=[]
)
