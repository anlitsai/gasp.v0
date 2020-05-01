#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 04:48:28 2019

@author: altsai
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

generate master bias, master dark, master flat for one month.
$ condaa
$ python slt_calibration_science_target.py slt201908
or
$ python slt_calibration_science_target.py slt20190822

"""



#dir_root='/home/altsai/project/20190801.NCU.EDEN/data/gasp/'
#dir_root='/home/altsai/gasp/lulin_data/2019/slt/'
#dir_month='slt201908'
#date=dir_month+'22'
#dir_master=dir_month+'_master/'
#dir_calib_sci=date+'_calib_sci/'


import os
import sys
import shutil
#import re
import numpy as np
#import numpy
from astropy.io import fits
#import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#import glob
#import subprocess
#from scipy import interpolate
#from scipy import stats
#from scipy.interpolate import griddata
#from time import gmtime, strftime
#import pandas as pd
from datetime import datetime
#from scipy import interpolate
#from scipy import stats
from astropy import units as u
from astropy.coordinates import SkyCoord
# https://docs.astropy.org/en/stable/coordinates/

print('-----------------')
print('input parameters')
print('-----------------')



#folder=sys.argv[1]
#folder='slt20190822'
folder=input("Enter Folder (ex: slt2019xxxx): ")

dir_month=folder[0:9]
date_obs=folder[3:11]
#print(dir_month)
#dir_master=dir_month+'_master/'
#print(dir_master)
dir_calib_sci=folder+'_calib_sci/'
#print(dir_calib_sci)

file_info=folder+'_target_info.txt'
if os.path.exists(file_info):
    os.remove(file_info)
f_info=open(file_info,'w')

file_log=folder+'_sci.log'
if os.path.exists(file_log):
    os.remove(file_log)
f_log=open(file_log,'w')

print(sys.argv)
f_log.write(str(print(sys.argv)))

info_folder='Your folder is :'+str(folder)
f_log.write(info_folder+'\n')


'''
if os.path.exists(dir_calib_sci):
    shutil.rmtree(dir_calib_sci)
os.makedirs(dir_calib_sci,exist_ok=True)
'''

'''
if os.path.exists(dir_master):
    shutil.rmtree(dir_master)
os.makedirs(dir_master,exist_ok=True)

print('...generate master files on '+dir_month+'...')
'''
#sys.exit(0)

'''
logfile=dir_month+'_master.log'
sys.stdout=open(logfile,'w')
print(sys.argv)
'''


par1=10.
par2=3.
par3=12



#time_calib_start=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#time_calib_start=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
time_calib_start=str(datetime.now())  
info_time=('Data calibrated by An-Li Tsai at '+time_calib_start+' UTC')
print(info_time)
f_log.write(info_time+'\n')

print(' ---------------------------')
print(' Science Target ')
print(' ---------------------------')
f_log.write(' ---------------------------\n')
f_log.write(' Science Target \n')
f_log.write(' ---------------------------\n')


cmd_search_file_sci="find ./ | grep "+folder+" | grep fts | grep GASP |sort -t'@'"
f_log.write(cmd_search_file_sci+'\n')
list_file_sci=os.popen(cmd_search_file_sci,"r").read().splitlines()
info_print='...calibrating science targets...'
print(info_print)
f_log.write(info_print+'\n')
print(list_file_sci)
f_log.write(str(list_file_sci)+'\n')
n_file_sci=len(list_file_sci)
info_n_sci='... found '+str(n_file_sci)+' science targets ...'
print(info_n_sci)
f_log.write(info_n_sci+'\n')

#cmd_search_sci_filter="find ./ |grep "+date" | grep fts |grep GASP | cut -d / -f4 | awk -F'_Astro' '{print $1}'| rev | cut -c1-3 | rev | cut -d - -f2 "

#os.chdir(dir_root+"wchen/wchen_03_GASP_01/")

#cmd_sci1="ls ./ | awk -F'_Astrodon' '{print $1}'| awk '{print substr($0,length-2,3)}' | cut -d - -f2 |sort |uniq"
#print(sci_filter_list)

#sci_list=os.popen("ls","r").read().splitlines()
#print(sci_list)

#calib_sci={}

head_info='Index | Date | Filename                                  | Object | RA_hhmmss | DEC_ddmmss | RA_deg            | DEC_deg          | FilterName      | JD(day)        | ExpTime(sec) | ZMAG(mag)  | FWHM'
f_info.write(head_info+'\n')

k=0
for i in list_file_sci:
    k=k+1
    idx=str(k)
    filename_sci=[i.split('/',-1)[-1]][0]
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
#    date_obs=imhead['DATE-OBS']
#    time_obs=imhead['TIME-OBS']
    jd=imhead['JD']
    obj=imhead['OBJECT']
    fwhm=imhead['FWHM']
    zmag=imhead['ZMAG']
    ra_hhmmss=imhead['RA']
    dec_ddmmss=imhead['Dec']
    radec_deg=SkyCoord(ra_hhmmss,dec_ddmmss,unit=(u.hourangle,u.deg))
#    ra_deg=SkyCoord(ra_hhmmss,unit=(u.hourangle))
#    dec_deg=SkyCoord(dec_ddmmss,unit=(u.deg))
    ra_deg=radec_deg.ra.deg
    dec_deg=radec_deg.dec.deg
    print(ra_deg,dec_deg)
    filter_name=imhead['FILTER']
#    select_master_dark=master_dark
    cmd_sci_filter='echo '+filter_name+' | cut -d _ -f1'
#    print(cmd_sci_filter)
    sci_filter=os.popen(cmd_sci_filter,"r").read().splitlines()[0]
#    print(sci_filter)
    idx_filter_time=sci_filter+"_"+idx_time
    info_sci=str(k)+' [DATE] '+date_obs+ str(filename_sci)+' [OBJ] '+str(obj)+' [RA_hhmmss] '+ra_hhmmss+' [DEC_ddmmss] '+dec_ddmmss+' [RA_deg] '+str(ra_deg)+' [DEC_deg] '+str(dec_deg)+' [FIL] '+filter_name+' [JD] '+str(jd)+' [EXPTIME] '+str(exptime)+' [ZMAG] '+str(zmag)+' [FWHM] '+str(fwhm)
#    print(info_sci)
    f_log.write(info_sci+'\n')
    info_write=str(idx)+' | '+date_obs+' | '+ str(filename_sci)+' | '+str(obj)+' | '+ra_hhmmss+' | '+dec_ddmmss+' | '+str(ra_deg)+' | '+str(dec_deg)+' | '+filter_name+' | '+str(jd)+' | '+str(exptime)+' | '+str(zmag)+' | '+str(fwhm)
    f_info.write(info_write+'\n')
#    print(idx_filter_time)
#    select_master_flat=master_flat[sci_filter]
#    print(select_master_flat[1000][1000])
#    select_master_dark=master_dark[idx_time]
#    print(select_master_dark[1000][1000])
#    sci_flat=(imdata-master_bias-select_master_dark)/select_master_flat
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
#    print('...output calibrated '+sci_name+' to fits file...')
#    time_calib=str(datetime.now())  
#    imhead.add_history('Master bias, dark, flat are applied at '+time_calib+' UTC+8 by An-Li Tsai')
#    fitsname_calib_sci=sci_name+'_calib.fits'
    #hdu=fits.PrimaryHDU(calib_sci[i])
#    fits.writeto(fitsname_calib_sci,data=sci_flat,header=imhead,overwrite=True)

f_info.close()
f_log.close()
    
print('...grap header information from '+folder+' ... done...')
