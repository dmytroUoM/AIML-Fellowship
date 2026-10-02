$v = 'C:\AI_ML_Fellowship\Project2\.venv-tomo'

# 1. What the venv was created from
Get-Content "$v\pyvenv.cfg"

# 2. Decisive: compiled-extension filenames carry the Python tag (cp311 vs cp313)
Get-ChildItem "$v\Lib\site-packages\numpy\core" -Filter '_multiarray_umath*.pyd' | Select Name
Get-ChildItem "$v\Lib\site-packages\torch\lib" -Filter 'torch_cpu.dll' | Select Name, Length
Get-Content (Get-ChildItem "$v\Lib\site-packages" -Filter 'torch-*.dist-info').FullName\WHEEL

# 3. What actually runs
& "$v\Scripts\python.exe" -c "import sys,torch,numpy,scipy,cv2; print(sys.version); print(torch.__version__, numpy.__version__, scipy.__version__, cv2.__version__)"

# 4. Whether the venv's installed set matches the freeze
& "$v\Scripts\python.exe" -m pip freeze | Out-File D:\freeze_now.txt
Compare-Object (Get-Content C:\AI_ML_Fellowship\Project2\requirements.txt) (Get-Content D:\freeze_now.txt)

# 5. What --no-deps may have hidden
& "$v\Scripts\python.exe" -m pip check

# 6. Recover your own command history
Select-String -Path (Get-PSReadLineOption).HistorySavePath -Pattern 'pip install|venv|py -3'