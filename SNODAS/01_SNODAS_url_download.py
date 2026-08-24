#imports
import requests
import os
import glob
import tarfile


dir = "."

urls = ["https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2023/04_Apr/SNODAS_20230405.tar",
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2024/03_Mar/SNODAS_20240315.tar",
]

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


#From here, TAR files can be converted to geoTIFF using command line arguments found here: https://nsidc.org/data/user-resources/help-center/how-do-i-convert-snodas-binary-files-geotiff-or-netcdf


