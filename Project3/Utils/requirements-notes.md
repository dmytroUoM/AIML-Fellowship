# requirements-notes.md
# freeze 
py -m pip freeze > requirements.txt

1. py -m pip install -r requirements.txt --no-deps
2. Verify: py -c "import torch, numpy; print(torch.from_numpy(numpy.zeros(3)))"
   (should print a tensor with no warning - confirms numpy/torch ABI match)
   
py -3.11 -m venv .venv-project3
.\.venv-project3\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r .\Utils\requirements.txt

pip freeze > requirements.lock.txt