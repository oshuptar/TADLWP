# Faculty computers

### Preparation: create venv, install libraries (only once)

Extract the zip to a directory of your choice. Create a virtual environment (if you don't have it from the previous labs) and install the required packages. You can use the same venv for all laboratories.

```
python -m venv .venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
source .venv/bin/activate
```

### Launch code-oss

For jupyter notebook kernels to be discoverable, open the IDE in the folling way:

```
code-oss --enable-proposed-api ms-toolsai.jupyter
```
