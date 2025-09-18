from huggingface_hub import hf_hub_download
from huggingface_hub import login
import subprocess
import os

login("")
location = "./model/new/"
m1_path = hf_hub_download(
    repo_id="firdavsus/navoist_KD_faster",
    filename="model.bin",
    cache_dir=location
)

m2_path = hf_hub_download(
    repo_id="firdavsus/navoist_KD_faster",
    filename="vocabulary.json",
    cache_dir=location
) 

m3_path = hf_hub_download(
    repo_id="firdavsus/navoist_KD_faster",
    filename="config.json",
    cache_dir=location
)

print(os.listdir())
subprocess.run(["cp", m1_path, "model/faster-go/"])
subprocess.run(["cp", m2_path, "model/faster-go/"])
subprocess.run(["cp", m3_path, "model/faster-go/"])
