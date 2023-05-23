# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:24:38 2023

@author: rmfha
"""

import h5py
import numpy as np 

files_check_IS2 = !ls E:/DTU/CRYO2ICE/Data/CRYO2ICE_data/IS2/All_data/*_01.h5

#filename = r"E:\DTU\CRYO2ICE\Data\CRYO2ICE_data\IS2\All_data\ATL10-01_20210412125041_02861101_005_01.h5"

for i in files_check_IS2:
    with h5py.File(i, "r") as f:
    start_IS2_UTC = f['ancillary_data/data_start_utc'][:]
    stop_IS2_UTC = f['ancillary_data/data_end_utc'][:]
    IS2_RGT = f['ancillary_data/end_rgt'][:]
    
    print(start_IS2_UTC, stop_IS2_UTC, IS2_RGT)
