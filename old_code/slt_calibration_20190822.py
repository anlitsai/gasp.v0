#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 03:11:27 2019

@author: altsai
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

data calibration for slt20190822.
"""

dir_root='/home/altsai/project/20190801.NCU.EDEN/data/gasp/'
#dir_root='/home/altsai/gasp/lulin_data/2019/slt/'
month='slt201908'
date=month+'22'
dir_master=month+'_master/'
dir_calib_sci=date+'_calib_sci/'


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

logfile='slt20190822.log'
sys.stdout=open(logfile,'w')
print(sys.argv)

#time_calib_start=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#time_calib_start=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
time_calib_start=str(datetime.now())  
print('Data calibrated by An-Li Tsai at '+time_calib_start+' UTC')


#shutil.rmtree(dir_master)
#shutil.rmtree(dir_calib_sci)
os.makedirs(dir_master,exist_ok=True)
os.makedirs(dir_calib_sci,exist_ok=True)

parm=3

'''
def reject_outliers(data,m=2.):
    diff=np.abs(data-np.median(data))
    mdev=np.median(diff)
    s=diff/mdev if mdev else 0
    return data[s<m]

print(reject_outliers(array_each_bias))
'''


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

cmd_search_file_bias='find ./ |grep '+month+' | grep fts | grep Bias'
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

fitsname_master_bias='master_bias_'+month+'.fits'
hdu=fits.PrimaryHDU(master_bias)
hdu.writeto(fitsname_master_bias,overwrite=True)
#now=str(datetime.now())  
#imhead.add_history('Master bias is generated at '+now+' UTC')
#fits.writeto(fitsname_master_bias,data=master_bias,header=imhead,overwrite=True)


#sys.exit(0)

print(' ---------------------------')
print(' Master Dark (subtract from Bias for different expotime) ')
print(' ---------------------------')
#cmd_search_file_dark='find ./ |grep '+month+' | grep fts | grep Dark'
#list_file_dark=os.popen(cmd_search_file_dark,"r").read().splitlines()
#print(list_file_dark)

cmd_search_dark_time='find ./ |grep '+month+' | grep fts | grep Dark | cut -d / -f4 | cut -d - -f3 | cut -d . -f1 | sort | uniq'
list_dark_time=os.popen(cmd_search_dark_time,"r").read().splitlines()
#print(list_dark_time)

#sys.exit(0)

print('...start to remove outlier dark')
bias_keep=reject_outliers_2x2img(array_each_bias,parm)
#print(bias_keep)

master_dark={}

for i in list_dark_time:
    cmd_search_file_dark_time='find ./ |grep '+month+' | grep fts | grep Dark | grep '+i
    list_file_dark_time=os.popen(cmd_search_file_dark_time,"r").read().splitlines()
    #array_each_dark_time=np.array([pyfits.getdata(j) for j in list_file_dark_time])
    array_each_dark_time=np.array([fits.open(j)[0].data for j in list_file_dark_time])
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
    fitsname_master_dark='master_dark_'+i+'_'+month+'.fits'
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

cmd_search_flat_filter="find ./ |grep "+month+" |grep fts | grep AutoFlat | cut -d / -f4 | awk -F'_Astro' '{print $1}'| rev | cut -c1-3 | rev | cut -d - -f2 | sort | uniq"
list_flat_filter=os.popen(cmd_search_flat_filter,"r").read().splitlines()
#print(list_flat_filter)

#sys.exit(0)

master_flat={}
#print(master_flat)

for i in list_flat_filter:
    cmd_search_file_flat_filter='find ./ |grep '+month+' | grep fts | grep AutoFlat | grep '+i+'_Astrodon'
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
        fitsname_master_flat='master_flat_'+idx_filter_time+'_'+month+'.fits'
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

#sys.exit(0)
#os.chdir
#print(master_flat)
#print(master_flat['R_180S'])
print(' ---------------------------')
print(' Science Target ')
print(' ---------------------------')

print(date)
cmd_search_file_sci="find ./ | grep "+date+" | grep fts | grep GASP "
list_file_sci=os.popen(cmd_search_file_sci,"r").read().splitlines()
print('...calibrating science targets...')
print(list_file_sci)

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
    sci_flat=imdata/select_master_flat
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
    imhead.add_history('Master bias, dark, flat are applied at '+time_calib+' UTC by An-Li Tsai')
    fitsname_calib_sci=sci_name+'_calib_sci.fits'
    #hdu=fits.PrimaryHDU(calib_sci[i])
    fits.writeto(fitsname_calib_sci,data=sci_flat,header=imhead,overwrite=True)

    
#sys.exit(0)

print('...move calibrated science files to '+dir_calib_sci+'...')

#cmd_search_master_files="find ./ -type f -name 'master*'"
cmd_search_calib_sci_files="find ./ |grep calib_sci"
list_calib_sci_files=os.popen(cmd_search_calib_sci_files,"r").read().splitlines()
#print(list_calib_sci_files)

for i in list_calib_sci_files:
    cmd_mv_file="mv "+i+" ./"+dir_calib_sci
    list_calib_sci_files=os.popen(cmd_mv_file,"r")



time_calib=str(datetime.now())
print('...finish calibration at '+time_calib+' UTC...')
