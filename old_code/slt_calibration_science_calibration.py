#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:54:00 2019

@author: altsai
"""

"""
Spyder Editor

data calibration for science target.
$ condaa
$ python slt_calibration_science_target.py.py _FOLDER_NAME_

for example:
$ python slt_calibration_science_target.py.py slt20190822
"""

dir_root='/home/altsai/project/20190801.NCU.EDEN/data/gasp/'
#dir_root='/home/altsai/gasp/lulin_data/2019/slt/'

#folder='slt201908'
#date=month+'22'
#dir_master=folder+'_master/'
#dir_calib_sci=date+'_calib_sci/'

#print(month,date)

import os
import sys
import shutil
#import re
import numpy as np
#import numpy
from astropy.io import fits
#import pyfits
import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#import glob
#import subprocess
#from scipy import interpolate
#from scipy import stats
#from scipy.interpolate import griddata
#from time import gmtime, strftime
#import pandas as pd
from datetime import datetime




folder=sys.argv[1]
#print(folder)
month=folder[3:9]
date=folder[-2:]
#print(month)
dir_master=folder[0:-2]+'_master/'
#print(dir_master)
dir_calib_sci=folder+'_calib_sci/'
#print(dir_calib_sci)

#shutil.rmtree(dir_calib_sci)
os.mkdir(dir_calib_sci)

print('...calibrate science target on '+folder+'...')

#sys.exit(0)

logfile=folder+'_sci.log'
sys.stdout=open(logfile,'w')
print(sys.argv)

#time_calib_start=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#time_calib_start=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
time_calib_start=str(datetime.now())  
print('Data calibrated by An-Li Tsai at '+time_calib_start+' UTC+8')
#print('')

print(' ---------------------------')
print(' Load Master Bias ')
print(' ---------------------------')

cmd_search_file_bias='find ./'+dir_master+' | grep fits | grep master_bias'
print(cmd_search_file_bias)
file_bias=os.popen(cmd_search_file_bias,"r").read().splitlines()[0]
#print(file_bias)
#print('len(file_bias)')
#print(len(file_bias))
#sys.exit(0)

#array_each_bias=np.array([pyfits.getdata(i) for i in file_bias])
master_bias=fits.open(file_bias)[0].data
#print(master_bias)
print('...load master bias: '+file_bias+'...')

#sys.exit(0)

print(' ---------------------------')
print(' Load Master Dark ')
print(' ---------------------------')

master_dark={}

cmd_search_file_dark='find ./'+dir_master+' | grep fits | grep master_dark'
#print(cmd_search_file_dark)
list_file_dark=os.popen(cmd_search_file_dark,"r").read().splitlines()
#print(list_file_dark)

for i in list_file_dark:
#    print(i)
    cmd_filename_dark='echo '+i+' | cut -d / -f3'
    filename_master_dark=os.popen(cmd_filename_dark,"r").read().splitlines()[0]
    print('...load master dark file: '+filename_master_dark+'...')
    cmd_idx_dark_time='echo '+i+' | cut -d / -f3 | cut -d _ -f3'
#    print(cmd_idx_dark_time)
    idx_master_dark_time=os.popen(cmd_idx_dark_time,"r").read().splitlines()[0]
#    print(idx_master_dark_time)
    data=fits.open(i)[0].data
#    print(data)
    master_dark[idx_master_dark_time]=data
#    print('--------')
    
#print('300S')
#print(master_dark['300S'])

#sys.exit(0)

print(' ---------------------------')
print(' Load Master Flat ')
print(' ---------------------------')

master_flat={}

cmd_search_file_flat='find ./'+dir_master+' | grep fits | grep master_flat'
#print(cmd_search_file_flat)
list_file_flat=os.popen(cmd_search_file_flat,"r").read().splitlines()
#print(list_file_flat)

for i in list_file_flat:
#    print(i)
    cmd_filename_flat='echo '+i+' | cut -d / -f3 '
#    print(cmd_idx_flat_filter)
    filename_flat=os.popen(cmd_filename_flat,"r").read().splitlines()[0]
    print('...load master flat file: '+filename_flat+'...')
    cmd_idx_flat_filter='echo '+i+' | cut -d / -f3 | cut -d _ -f3-4'
#    print(cmd_idx_flat_filter)
    idx_master_flat_filter=os.popen(cmd_idx_flat_filter,"r").read().splitlines()[0]
#    print(idx_master_flat_filter)
    data=fits.open(i)[0].data
#    print(data)
    master_flat[idx_master_flat_filter]=data
#    print('--------')
    
#print('gp_015S')
#print(master_flat['gp_015S'])

#sys.exit(0)

print(' ---------------------------')
print(' Science Target ')
print(' ---------------------------')

print(folder)
cmd_search_file_sci="find ./ | grep "+folder+" | grep fts | grep GASP "
list_file_sci=os.popen(cmd_search_file_sci,"r").read().splitlines()
print('...calibrating science targets...')
print(list_file_sci)
print(len(list_file_sci))

#cmd_search_sci_filter="find ./ |grep "+date" | grep fts |grep GASP | cut -d / -f4 | awk -F'_Astro' '{print $1}'| rev | cut -c1-3 | rev | cut -d - -f2 "

#os.chdir(dir_root+"wchen/wchen_03_GASP_01/")

#cmd_sci1="ls ./ | awk -F'_Astrodon' '{print $1}'| awk '{print substr($0,length-2,3)}' | cut -d - -f2 |sort |uniq"
#print(sci_filter_list)

#sci_list=os.popen("ls","r").read().splitlines()
#print(sci_list)

#calib_sci={}

for i in list_file_sci:
    hdu=fits.open(i)[0]
    imhead=hdu.header
    imdata=hdu.data
#    print(imdata.shape)
    exptime=imhead['EXPTIME']
    idx_time=str(int(exptime))+'S'
#    print(idx_time)
#    print(exptime)
#    naxis=imhead['NAXIS']
#    print(naxis)
    jd=imhead['JD']
    obj=imhead['OBJECT']
    fwhm=imhead['FWHM']
    zmag=imhead['ZMAG']
    ra=imhead['RA']
    dec=imhead['Dec']
    filter_name=imhead['FILTER']
    select_master_dark=master_dark[idx_time]
    cmd_sci_filter='echo '+filter_name+' | cut -d _ -f1'
#    print(cmd_sci_filter)
    sci_filter=os.popen(cmd_sci_filter,"r").read().splitlines()[0]
#    print(sci_filter)
    idx_filter_time=sci_filter+"_"+idx_time
#    print(idx_filter_time)
    select_master_flat=master_flat[idx_filter_time]
#    print(select_master_flat[1000][1000])
#    select_master_dark=master_dark[idx_time]
#    print(select_master_dark[1000][1000])
    sci_flat=(imdata-master_bias-select_master_dark)/select_master_flat
    #calib_sci[i]=sci_flat
#    print(time_idx,sci_filter)
#    print(time_idx,sci_filter)
#    print(select_master_flat.shape)
#    print(select_master_dark.shape)
#    print(sci_flat.shape)
    cmd_sci_name='echo '+i+' | cut -d / -f5 | cut -d . -f1'
#    print(cmd_sci_name)
    sci_name=os.popen(cmd_sci_name,"r").read().splitlines()[0]
#    print(sci_name)
#    plt.title(sci_name)
#    plt.imshow(sci_flat,cmap='rainbow')
#    plt.show()
    print('...output calibrated '+sci_name+' to fits file...')
    time_calib=str(datetime.now())  
    imhead.add_history('Master bias, dark, flat are applied at '+time_calib+' UTC+8 by An-Li Tsai')
    fitsname_calib_sci=sci_name+'_calib.fits'
    #hdu=fits.PrimaryHDU(calib_sci[i])
    fits.writeto(fitsname_calib_sci,data=sci_flat,header=imhead,overwrite=True)

    
#sys.exit(0)

print('...move calibrated science files to '+dir_calib_sci+'...')

#cmd_search_master_files="find ./ -type f -name 'master*'"
cmd_search_calib_sci_files="find ./ |grep calib.fits | grep "+date+"@"
list_calib_sci_files=os.popen(cmd_search_calib_sci_files,"r").read().splitlines()
print(list_calib_sci_files)
print(len(list_calib_sci_files))

for i in list_calib_sci_files:
    cmd_mv_file="mv "+i+" ./"+dir_calib_sci
    list_calib_sci_files=os.popen(cmd_mv_file,"r")


print(' ---------------------------')

time_calib=str(datetime.now())
print('...finish calibration at '+time_calib+' UTC+8...')


