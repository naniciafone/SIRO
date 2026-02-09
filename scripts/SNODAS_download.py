#imports
import requests
import os
import glob
import tarfile


dir = "."

urls = ["https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2023/04_Apr/SNODAS_20230405.tar",
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2024/03_Mar/SNODAS_20240315.tar",
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2024/04_Apr/SNODAS_20240418.tar",
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2025/04_Apr/SNODAS_20250404.tar",
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2025/05_May/SNODAS_20250501.tar"]

for url in urls:
    filename = os.path.basename(url)
    if os.path.exists(filename):
        continue

    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)


files = glob.glob(os.path.join(dir, "*.tar"))
for file in files:
    with tarfile.open(file) as tar:
        tar.extractall(dir)