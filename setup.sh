#!/bin/sh

virtualenv -p /usr/bin/python3.8 venv 
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download 'en_core_web_sm'
