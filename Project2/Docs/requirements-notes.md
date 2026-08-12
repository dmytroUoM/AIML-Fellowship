# requirements-notes.md
# freeze 
py -m pip freeze > requirements.txt

1. py -m pip install -r requirements.txt --no-deps
2. Verify: py -c "import torch, numpy; print(torch.from_numpy(numpy.zeros(3)))"
   (should print a tensor with no warning - confirms numpy/torch ABI match)