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



dir_root='/home/altsai/project/20190801.NCU.EDEN/data/gasp/'
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




#folder=sys.argv[1]
folder='slt201908'
dir_month=folder[0:9]
#print(dir_month)
dir_master=dir_month+'_master/'
#print(dir_master)
#dir_calib_sci=date+'_calib_sci/'
#print(dir_calib_sci)

os.makedirs(dir_master,exist_ok=True)

print('...generate master files on '+dir_month+'...')

#sys.exit(0)

logfile=dir_month+'.log'
sys.stdout=open(logfile,'w')
print(sys.argv)


#time_calib_start=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#time_calib_start=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
time_calib_start=str(datetime.now())  
print('Data calibrated by An-Li Tsai at '+time_calib_start+' UTC')




'''
def reject_outliers(data,m=2.):
    diff=np.abs(data-np.median(data))
    mdev=np.median(diff)
    s=diff/mdev if mdev else 0
    return data[s<m]

print(reject_outliers(array_each_bias))
'''

parm=3

def reject_outliers_2x2img(data3d,threshold):
    median_value=np.median(np.mean(data3d,axis=(1,2)))
    #print(median_value)
    diff=abs(data3d-median_value)
    #print(diff)
    med_diff=np.median(diff)
    #print(med_diff)
#    print('...remove outlier...')
    if med_diff == 0:
        s=0
    else:
        s=diff/med_diff
        #print(s)
    data3d_keep=np.where(s<threshold,data3d,np.NaN)
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data3d_keep

#array_each_bias_keep=array_each_bias[(np.abs(stats.zscore(array_each_bias)<3).all(axis=0))]
#array_each_bias_keep=array_each_bias[abs()]


print(' ---------------------------')
print(' Master Bias (mean) ')
print(' ---------------------------')

cmd_search_file_bias='find ./ |grep '+dir_month+' | grep fts | grep Bias'
list_file_bias=os.popen(cmd_search_file_bias,"r").read().splitlines()
#print(list_file_bias)
#print('len(list_file_bias)')
#print(len(list_file_bias))

#array_each_bias=np.array([pyfits.getdata(i) for i in list_file_bias])
array_each_bias=np.array([fits.open(i)[0].data for i in list_file_bias])
#print(array_each_bias.shape)

print('...start to remove outlier bias')
bias_keep=reject_outliers_2x2img(array_each_bias,parm)
#print(bias_keep)

print('...generate master bias...')
master_bias=np.nanmean(bias_keep, axis=0)
#print(master_bias)
#plt.title('Master Bias')
#plt.imshow(master_bias)
#plt.show()

print('...output master bias to fits file...')

fitsname_master_bias='master_bias_'+dir_month+'.fits'
hdu=fits.PrimaryHDU(master_bias)
hdu.writeto(fitsname_master_bias,overwrite=True)
#now=str(datetime.now())  
#imhead.add_history('Master bias is generated at '+now+' UTC')
#fits.writeto(fitsname_master_bias,data=master_bias,header=imhead,overwrite=True)


#sys.exit(0)

print(' ---------------------------')
print(' Master Dark (subtract from Bias for different expotime) ')
print(' ---------------------------')
#cmd_search_file_dark='find ./ |grep '+dir_month+' | grep fts | grep Dark'
#list_file_dark=os.popen(cmd_search_file_dark,"r").read().splitlines()
#print(list_file_dark)

cmd_search_dark_time='find ./ |grep '+dir_month+' | grep fts | grep Dark | cut -d / -f4 | cut -d - -f3 | cut -d . -f1 | sort | uniq'
list_dark_time=os.popen(cmd_search_dark_time,"r").read().splitlines()
#print(list_dark_time)

#sys.exit(0)

print('...start to remove outlier dark')
bias_keep=reject_outliers_2x2img(array_each_bias,parm)
#print(bias_keep)

master_dark={}

for i in list_dark_time:
    cmd_search_file_dark_time='find ./ |grep '+dir_month+' | grep fts | grep Dark | grep '+i
    list_file_dark_time=os.popen(cmd_search_file_dark_time,"r").read().splitlines()
    #array_each_dark_time=np.array([pyfits.getdata(j) for j in list_file_dark_time])
    array_each_dark_time=np.array([fits.open(j)[0].data for j in list_file_dark_time])
#    for j in list_file_dark_time:    
#        hdu=fits.open(j)[0]
#        data=hdu.data
#        data_arr=np.array(data)
#        array_each_dark_time[j][:][:]=np.array(data_arr[:][:])
    print('...remove outlier dark '+i+'...')
    dark_each_time_keep=reject_outliers_2x2img(array_each_dark_time,parm)
#    print(dark_keep)
    print('...generate master dark '+i+'...')
    master_dark_each_time=np.nanmean(dark_each_time_keep-master_bias,axis=0)
    master_dark[i]=master_dark_each_time
#    print(master_dark_each_time[1000][1000])
#    plt.title('Master Dark '+i)
#    plt.imshow(master_dark_each_time)
#    plt.show()
    print('...output master dark '+i+' to fits file...')
    fitsname_master_dark='master_dark_'+i+'_'+dir_month+'.fits'
    hdu=fits.PrimaryHDU(master_dark[i])
    hdu.writeto(fitsname_master_dark,overwrite=True)
#    now=str(datetime.now())  
#    imhead.add_history('Master bias is applied at '+now+' UTC')
#    fits.writeto(fitsname_master_dark,data=master_dark_each_time,header=imhead,overwrite=True)


#sys.exit(0)

print(' ---------------------------')
print(' Master Flat (subtract from Dark with different expotime for different filter) ')
print(' ---------------------------')

#os.chdir(dir_date+"/flat/")

cmd_search_flat_filter="find ./ |grep "+dir_month+" |grep fts | grep AutoFlat | cut -d / -f4 | awk -F'_Astro' '{print $1}'| rev | cut -c1-3 | rev | cut -d - -f2 | sort | uniq"
list_flat_filter=os.popen(cmd_search_flat_filter,"r").read().splitlines()
#print(list_flat_filter)

#sys.exit(0)

master_flat={}
#print(master_flat)

for i in list_flat_filter:
    cmd_search_file_flat_filter='find ./ |grep '+dir_month+' | grep fts | grep AutoFlat | grep '+i+'_Astrodon'
#    print(cmd_search_file_flat_filter)
    list_file_flat_filter=os.popen(cmd_search_file_flat_filter,"r").read().splitlines()
#    print(list_file_flat_filter)
#    print(len(list_file_flat_filter))
    #array_each_flat_filter=np.array([pyfits.getdata(j) for j in list_file_flat_filter])
    array_each_flat_filter=np.array([fits.open(j)[0].data for j in list_file_flat_filter])
#    print(array_each_flat_filter.shape)
    for j in list_dark_time:
        idx_filter_time=i+"_"+j
#        print(idx_filter_time)
#        print('...remove outlier flat '+idx_filter_time+'...')
#        array_each_flat_filter_keep=reject_outliers_2x2img(array_each_flat_filter,3)
        print('...generate master flat '+idx_filter_time+'...')
        mean_flat_each_filter=np.mean(array_each_flat_filter-master_bias-master_dark[j],axis=0)  
#        print(np.amax(mean_flat_each_filter))
        norm_mean_flat_each_filter=mean_flat_each_filter/np.nanmax(mean_flat_each_filter)
#        print(np.amax(norm_mean_flat_each_filter))
        master_flat[idx_filter_time]=norm_mean_flat_each_filter
#        print(master_flat[idx_filter_time])
#    print(mean_flat_each_filter[1000][1000])
#    plt.title('Master Flat '+i)
#    plt.imshow(mean_flat_each_filter)
#    plt.show()
        print('...output master flat '+idx_filter_time+' to fits file...')
        fitsname_master_flat='master_flat_'+idx_filter_time+'_'+dir_month+'.fits'
        hdu=fits.PrimaryHDU(master_flat[idx_filter_time])
        hdu.writeto(fitsname_master_flat,overwrite=True)
#        now=str(datetime.now())  
#        imhead.add_history('Master bias, dark are applied at '+now+' UTC')
#        fits.writeto(fitsname_master_flat,data=norm_mean_flat_each_filter,header=imhead,overwrite=True)



print('...move master files to '+dir_master+'...')
#cmd_search_master_files="find ./ -type f -name 'master*'"
cmd_search_master_files="find ./ |grep '^\./master_'"
list_master_files=os.popen(cmd_search_master_files,"r").read().splitlines()
#print(list_master_files)

for i in list_master_files:
    cmd_mv_file="mv "+i+" ./"+dir_master
    list_master_files=os.popen(cmd_mv_file,"r")
    
    