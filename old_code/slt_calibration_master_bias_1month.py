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
import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#import glob
#import subprocess
#from scipy import interpolate
#from scipy import stats
from scipy.interpolate import griddata
#from time import gmtime, strftime
#import pandas as pd
from datetime import datetime
from scipy import interpolate






#folder=sys.argv[1]
folder='slt201908'
dir_month=folder[0:9]
#print(dir_month)
dir_master=dir_month+'_master/'
#print(dir_master)
#dir_calib_sci=date+'_calib_sci/'
#print(dir_calib_sci)

shutil.rmtree(dir_master)
os.mkdir(dir_master)

print('...generate master files on '+dir_month+'...')

#sys.exit(0)

'''
logfile=dir_month+'.log'
sys.stdout=open(logfile,'w')
print(sys.argv)
'''

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

'''
def reject_outliers_img_file(data3d,threshold=3.):
    median_value=np.nanmedian(np.nanmedian(data3d,axis=(1,2)))
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
'''
#array_each_bias_keep=array_each_bias[(np.abs(stats.zscore(array_each_bias)<3).all(axis=0))]
#array_each_bias_keep=array_each_bias[abs()]


def reject_outliers_img_file(data3d,threshold):
    median_per_img=np.nanmedian(data3d,axis=(1,2))
    n=median_per_img.shape[0]
    median_value=np.nanmedian(median_per_img)
    #print(median_value)
    xx,yy,zz=data3d.shape
    data3d_keep=np.empty((xx,yy,zz))
    data3d_keep[:]=np.nan
    diff=abs(median_per_img-median_value)
    med_diff=np.median(diff)
    s=diff/med_diff
#    data3d_keep=np.where(s<threshold,data3d,np.nan)
    k=0
    for i in range(n):
        if med_diff == 0:
            s=0
        else:
            s=diff[i]/med_diff
            #print(s)
        if s<threshold:
            data3d_keep[i]=data3d[i]
            k=k+1
    bad=n-k
#    bad=np.count_nonzero(np.isnan(data3d_keep))
    rate=bad/n
    print('number of reject/total image: ',bad,'/',n,'=',rate)
#        data3d_keep[i]=np.where(s<threshold,data3d[i])
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data3d_keep




def reject_outliers_px(data2d,threshold):
    median_value=np.nanmedian(data2d)
    print('median:',median_value)
    diff=abs(data2d-median_value)
    print('data-median',diff)
    med_diff=np.median(diff)
    print('median(data-median)',med_diff)
#    print('...remove outlier...')
    s=diff/med_diff
    print('diff/med_diff',s)
    data2d_keep=np.where(s<threshold,data2d,np.NaN)
    bad=np.count_nonzero(np.isnan(data2d_keep))
    px_area=data2d.shape[0]*data2d.shape[1]
    rate=bad/px_area
    print('number of bad/total px: ',bad,'/',px_area,'=',rate)
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data2d_keep



def interp_per_img(data2d):
#    np.count_nonzero(~np.isnan(data))
    # 4194304=2048x2048
#    np.count_nonzero(np.isnan(data))
    # 0   
    x_grid = np.arange(0, data2d.shape[1])
    y_grid = np.arange(0, data2d.shape[0])
    #mask invalid values
    data_bad = np.ma.masked_invalid(data2d)
    xx, yy = np.meshgrid(x_grid, y_grid)
    #get only the valid values
    x1 = xx[~data_bad.mask]
    y1 = yy[~data_bad.mask]
    newdata = data_bad[~data_bad.mask]  
#    print(newdata)
    data_interp = interpolate.griddata((x1, y1), newdata.ravel(), (xx, yy), method='linear')
    return data_interp


par1=5
par2=10
par3=7

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
n_arr_bias=(array_each_bias.shape[0])


print('number of total px: 2048x2048x',n_arr_bias,' = ', 2048*2048*n_arr_bias)
count_nan=np.count_nonzero(np.isnan(array_each_bias))
print('number of bad px: ',count_nan)
count_value=np.count_nonzero(~np.isnan(array_each_bias))
print('number of good px: ',count_value)



print('...start to remove outlier bias...')
print('...remove outlier image...')
bias_keep=reject_outliers_img_file(array_each_bias,par1)
#print(bias_keep)

print('...generate master bias...')
#master_bias=np.nanmean(array_each_bias, axis=0)
#master_bias=np.nanmean(bias_keep, axis=0)
mean_bias=np.nanmean(bias_keep, axis=0)
#print(master_bias)
#plt.title('Master Bias')
#plt.imshow(master_bias)
#plt.show()

print('...remove outlier pixel...')
mean_bias_keep=reject_outliers_px(mean_bias,par2)



'''
print('number of px per img: 2048x2048 =',2048*2048)
count_nan=np.count_nonzero(np.isnan(master_bias))
print('number of bad px: ',count_nan)
count_value=np.count_nonzero(~np.isnan(master_bias))
print('number of good px: ',count_value)
'''

print('...interpolate bad pixel of master bias...')
master_bias=interp_per_img(mean_bias_keep)



'''
print('...start to remove outlier bias')
master_bias2=reject_outliers_per_img(master_bias,3)
#print(master_bias2)


print('number of px per img: 2048x2048 =',2048*2048)
count_nan=np.count_nonzero(np.isnan(master_bias2))
print('number of bad px: ',count_nan)
count_value=np.count_nonzero(~np.isnan(master_bias2))
print('number of good px: ',count_value)
'''


print('...output master bias to fits file...')

fitsname_master_bias='master_bias_'+dir_month+'.fits'
hdu=fits.PrimaryHDU(data=master_bias)
#hdr=fits.Header()
#now=str(datetime.now())  
#hdr.add_history('Master Bias generated at '+now+' UTC')
#hdu._writeheader('Master Bias generated at '+now+' UTC')
hdu.writeto(fitsname_master_bias,overwrite=True)
#now=str(datetime.now())  
#imhead.add_history('Master bias is generated at '+now+' UTC')
#fits.writeto(fitsname_master_bias,data=master_bias,header=imhead,overwrite=True)

