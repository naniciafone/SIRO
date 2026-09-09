#!/usr/bin/env python

# Script to download SPIReS-MODIS/Terra fSCA files from SnowToday on the FTP server
# Adapted from the NSIDC example: https://nsidc.org/data/user-resources/help-center/how-access-nsidc-data-using-ftp-client-command-line-wget-or-python

# NOTE: SPIRES CRS = ESRI:54008

from ftplib import FTP
import os
from tqdm import tqdm
import numpy as np

OUT_DIR = '/Users/rdcrlrka/Research/SIRO/MCS_SCA/SnowToday'

DATA_DIRS = [
    '/shares/snow-today/gridded_data/SPIRES_HIST_V01/h09v04/2022',
    '/shares/snow-today/gridded_data/SPIRES_HIST_V01/h09v04/2023',
    '/shares/snow-today/gridded_data/SPIRES_HIST_V01/h09v04/2024',
    '/shares/snow-today/gridded_data/SPIRES_HIST_V01/h09v04/2025'
]

MONTH_START = 10    # inclusive
MONTH_END = 6       # inclusive

# FTP server
ftpdir = 'dtn.rc.colorado.edu'

# Connect and log in to the FTP
print('Logging in')
ftp = FTP(ftpdir)
ftp.login('anonymous')

# Change to the destination directory on own computer where you want to save the files
os.chdir(OUT_DIR)

# Iterate over data directories
for data_dir in DATA_DIRS:

    # Change to the directory where the files are on the FTP
    print('\nChanging to '+ data_dir)
    ftp.cwd(data_dir)

    # Get a list of the files in the FTP directory
    files = ftp.nlst()

    # Filter by month
    dates = [x.split('_')[4] for x in files]
    dts = np.array([np.datetime64(f"{x[0:4]}-{x[4:6]}-{x[6:]}") for x in dates])
    months = dts.astype('datetime64[M]').astype(int) % 12 + 1
    ikeep = [i for i in range(len(months)) if (months[i] >= MONTH_START) or (months[i] <= MONTH_END)]
    files_filt = [files[i] for i in ikeep]

    # Download all the files within the FTP directory
    print("Downloading files...")
    for file in tqdm(files_filt):
        # Skip download if it already exists
        if os.path.exists(os.path.join(OUT_DIR, file)): 
            continue
        ftp.retrbinary('RETR ' + file, open(file, 'wb').write)
    
# Close the FTP connection
ftp.quit()