# Faculty computers

### Preparation: create venv, install libraries (only once)

Extract the zip to a directory of your choice. Create a virtual environment (if you don't have it from the previous labs) and install the required packages. Feel free to use other virtual environments (e.g. miniconda). Install in your default python is also possible, but not recommended if you're planning to use it for anything else. The same venv should work for all laboratories; you **don't** have to create it every time. Replace the index-url depending on your GPU type (https://pytorch.org/get-started/locally/). Make sure to install CUDA first, if you don't have it.

```
python -m venv .venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
source .venv/bin/activate
```

To check if the gpu was discovered by pytorch, run
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Solve the labs in the IDE of your choice.
