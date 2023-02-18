import subprocess

subprocess.run("uvicorn modules.app:app --timeout-keep-alive 300 --host 0.0.0.0 --port 7860", shell=True)
