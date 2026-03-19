# Google colab

```
!unzip -d /content/Lab2 -o /content/Lab2_student.zip
print("Successfully unzipped the lab files")
```

### Preparation: paste into the first cell. Change the zip path if needed

```
import sys

# Check if we are in a Python 3.12+ environment (where 'imp' is gone)
if sys.version_info >= (3, 12):
    try:
        import imp
    except ImportError:
        # Create a dummy 'imp' module to satisfy older IPython extensions
        import types
        import importlib.util
        
        imp_module = types.ModuleType("imp")
        imp_module.reload = importlib.reload
        # Add other common 'imp' aliases if needed, but 'reload' is the big one
        sys.modules["imp"] = imp_module
        print("Successfully patched 'imp' for Python 3.12 compatibility.")

# Now the magic command should work without crashing
%load_ext autoreload
%autoreload 2
print("Successfully activated .py files autoreload")

%matplotlib inline

import os
os.chdir('/content/Lab2/student/code')
print("Successfully changed the root directory")

!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
print("Successfully installed cloudflared for port tunneling")

!pip install -r ../requirements.txt
```


### You can now import and run python modules

```
from part_1 import part1

part1()
```


### Running streamlit in the background
```
import subprocess, time, re, os, signal

# Kill any previous instances
def cleanup_previous():
    for name in ["streamlit", "cloudflared"]:
        result = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Killed {name} (PID {pid})")
            except ProcessLookupError:
                pass
    time.sleep(1)  # Give processes time to die

cleanup_previous()

# Start both processes in background
subprocess.Popen(
    "nohup streamlit run network_dashboard.py > streamlit.log 2>&1 &"
    " nohup ./cloudflared tunnel --url http://localhost:8501 > tunnel.log 2>&1 &",
    shell=True
)

# Wait for tunnel URL to appear in logs
print("Waiting for tunnel URL...")
for _ in range(30):
    time.sleep(2)
    try:
        log = open("tunnel.log").read()
        match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", log)
        if match:
            print(f"Dashboard live at: {match.group(0)}")
            print(
                "You can now continue running the notebook, "
                "while the streamlit dashboard is active in the background."
            )
            break
    except FileNotFoundError:
        pass
else:
    print("Tunnel URL not found. Check tunnel.log")
```

### Running streamlit synchronously
```
!streamlit run network_dashboard.py & ./cloudflared tunnel --url http://localhost:8501
```
