import os
import sys
import subprocess


print("Getting rustup")
subprocess.run(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    shell=True,
)
print("Got rustup")
myenv = os.environ.copy()
myenv["PATH"] = os.path.expanduser("~/.cargo/bin:") + myenv["PATH"]
print("RUSTC", os.path.isfile(os.path.expanduser("~/.cargo/bin/rustc")))
subprocess.run("rustc --version", shell=True, env=myenv)
subprocess.run(
    "pip install -e git+https://github.com/huggingface/tokenizers/#egg=tokenizers\&subdirectory=bindings/python",
    shell=True,
    env=myenv,
)
sys.path.append(
    os.path.join(os.getcwd(), "src", "tokenizers", "bindings", "python", "py_src")
)


import tokenizers
