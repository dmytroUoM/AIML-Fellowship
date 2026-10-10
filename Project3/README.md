# AIML-Fellowship Project 3


python -m venv .venv-project3
.\.venv-project3\Scripts\Activate.ps1
py -m pip install -r requirements.txt --no-deps

py -3.11 -m venv .venv-project3
.\.venv-project3\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r .\Utilites\requirements.txt

pip freeze > requirements.lock.txt

