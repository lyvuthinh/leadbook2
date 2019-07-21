# leadbook2 Setup Guide:

## 1. Virtual Environment!

## 1.1 Upgrading/Setting up python-dev and pip for Ubuntu
```
sudo apt-get install python3-pip
sudo apt-get install python3-dev
pip3 install -U pip setuptools
```

In case of comflicting pip3 and current pip2 of ubuntu, just reinstall pip3:
python3 -m pip uninstall pip && sudo apt install python3-pip --reinstall

## 1.2 Setup virtual environment for quest
```
sudo pip3 install virtualenvwrapper

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3  
source `which virtualenvwrapper.sh`  

echo 'source /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc
bash
mkvirtualenv leadbook2
```
To activate virtual environment: `workon leadbook2`

To deactivate: `deactivate`

Then run 'pip3 install -r requirements.txt'
This will download and install the python packages in 2.

## 2. Django, MongoEngine and Python packages
```
pip install -U nltk
pip install pandas
pip install ipython
pip install ipython
pip install -U scikit-learn
```

download data of ntlk
import nltk
nltk.download('stopwords')
nltk.download('punkt')

## 3. Execution
1) Create the training data from raw data
```
workon leadbook2
python preprocess.py
```
We use a simple rule to create training data. For each raw job title, if it matches to any keyword in the departments.json, then it is classified to that department. By this rule, we have a handful of traing data to work with.


