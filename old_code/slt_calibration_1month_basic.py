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
import os
import sys
import re
import numpy as np
#import numpy
import pyfits
import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#from astropy.io import fits as pyfits
from astropy.io import fits
import glob
import subprocess

print(sys.argv)
#os.system("df -h")
#os.mkdir("test")
#os.rmdir("test")


#list_files=os.listdir("./bias-dark/")
#print(list_files)
#list_bias=os.system("ls ./bias-dark/ |grep Bias")
#list_bias=os.system("ls ./bias-dark/Bias*")
#print("---")
#list_bias=[f for f in os.listdir('./bias-dark/') if re.match('Bias',f)]
#print(list_bias)
#print("---")
#list_drak=[f for f in os.listdir('./bias-dark/') if re.match('Dark',f)]
#print(list_drak)
#list_dark=os.system("list=`find ./bias-dark/ |grep Dark`")
#print(list)
#print("---")






print(' ---------------------------')
print(' Master Bias (mean) ')
print(' ---------------------------')
dir_root='/home/altsai/project/20190801.NCU.EDEN/data/gasp/'
#dir_root='/home/altsai/gasp/lulin_data/2019/slt/'
month1='slt201908'

cmd_list_file_bias='find ./ |grep '+month1+' | grep fts | grep Bias'
list_file_bias=os.popen(cmd_list_file_bias,"r").read().splitlines()
#print(list_file_bias)
#print(len(list_file_bias))

#sys.exit(0)

bias=np.array([pyfits.getdata(i) for i in list_file_bias])
#print(np.amax(bias))
#print(np.amin(bias))
master_bias=np.mean(bias,axis=0)
#print(master_bias)
#print(np.mean(master_bias))
#print(np.shape(master_bias))
#print(np.where(master_bias>np.mean(master_bias)*1.1,master_bias,0))
#plt.title('Master Bias')
#plt.imshow(master_bias)
#plt.show()

#sys.exit(0)

print(' ---------------------------')
print(' Master Dark (subtract from Bias for different expotime) ')
print(' ---------------------------')
#cmd_list_file_dark='find ./ |grep '+month1+' | grep fts | grep Dark'
#list_file_dark=os.popen(cmd_list_file_dark,"r").read().splitlines()
#print(list_file_dark)

cmd_list_dark_time='find ./ |grep '+month1+' | grep fts | grep Dark | cut -d / -f4 | cut -d - -f3 | cut -d . -f1 | sort | uniq'
list_dark_time=os.popen(cmd_list_dark_time,"r").read().splitlines()
#print(list_dark_time)



#sys.exit(0)

master_dark={}

for i in list_dark_time:
    cmd_list_file_dark_time='find ./ |grep '+month1+' | grep fts | grep Dark | grep '+i
    list_file_dark_time=os.popen(cmd_list_file_dark_time,"r").read().splitlines()
#    print(len(list_file_dark_time))
    array_each_dark_time=np.array([pyfits.getdata(j) for j in list_file_dark_time])
#    print(array_each_dark_time.shape)
#    print(np.amax(array_each_dark_time))
    master_dark_each_time=np.mean(array_each_dark_time-master_bias,axis=0)
#    print(np.amax(master_dark_each_time))
#    print(np.amin(master_dark_each_time))
#    print(np.mean(master_dark_each_time))
    master_dark[i]=master_dark_each_time
#    print(np.where(master_dark_each_time>np.mean(master_dark_each_time)*1.5,master_dark_each_time,0))
#    print(master_dark_each_time[1000][1000])
#    plt.title('Master Dark '+i)
#    plt.imshow(master_dark_each_time)
#    plt.show()

#sys.exit(0)

print(' ---------------------------')
print(' Master Flat (subtract from Dark with different expotime for different filter) ')
print(' ---------------------------')

#os.chdir(dir_date+"/flat/")

#cmd_flat='ls |cut -d - -f4 | cut -d _ -f1 | uniq'
#list_flat_filter=os.popen(cmd_flat,"r").read().splitlines()

#cmd_list_file_flat='find ./ |grep '+month1+' | grep fts | grep AutoFlat'
#list_file_flat=os.popen(cmd_list_file_flat,"r").read().splitlines()
#print(list_file_flat)

cmd_list_flat_filter="find ./ |grep "+month1+" |grep fts | grep AutoFlat | cut -d / -f4 | awk -F'_Astro' '{print $1}'| rev | cut -c1-3 | rev | cut -d - -f2 | sort | uniq"
list_flat_filter=os.popen(cmd_list_flat_filter,"r").read().splitlines()
#print(list_flat_filter)

#sys.exit(0)

master_flat={}
#print(master_flat)

for i in list_flat_filter:
    cmd_list_file_flat_filter='find ./ |grep '+month1+' | grep fts | grep AutoFlat | grep '+i+'_Astrodon'
#    print(cmd_list_file_flat_filter)
    list_file_flat_filter=os.popen(cmd_list_file_flat_filter,"r").read().splitlines()
#    print(list_file_flat_filter)
#    print(len(list_file_flat_filter))
    array_each_flat_filter=np.array([pyfits.getdata(j) for j in list_file_flat_filter])
#    print(array_each_flat_filter.shape)
    for j in list_dark_time:
        idx_filter_time=i+"_"+j
#        print(idx_filter_time)
        master_flat_each_filter=np.mean(array_each_flat_filter-master_dark[j],axis=0)  
#        print(np.amax(master_flat_each_filter))
        norm_master_flat_each_filter=master_flat_each_filter/np.amax(master_flat_each_filter)
#        print(np.amax(norm_master_flat_each_filter))
        master_flat[idx_filter_time]=norm_master_flat_each_filter
#        print(master_flat[idx_filter_time])
#    print(master_flat_each_filter[1000][1000])
#    plt.title('Master Flat '+i)
#    plt.imshow(master_flat_each_filter)
#    plt.show()

#sys.exit(0)

#print(master_flat)
#print(master_flat['R_180S'])
print(' ---------------------------')
print(' Science Target ')
print(' ---------------------------')

cmd_list_file_sci="find ./ |grep "+month1+" | grep fts | grep GASP "
list_file_sci=os.popen(cmd_list_file_sci,"r").read().splitlines()

cmd_list_sci_filter="find ./ |grep "+month1+" | grep fts |grep GASP | cut -d / -f4 | awk -F'_Astro' '{print $1}'| rev | cut -c1-3 | rev | cut -d - -f2 "

#os.chdir(dir_root+"wchen/wchen_03_GASP_01/")

#cmd_sci1="ls ./ | awk -F'_Astrodon' '{print $1}'| awk '{print substr($0,length-2,3)}' | cut -d - -f2 |sort |uniq"
#print(sci_filter_list)

#sci_list=os.popen("ls","r").read().splitlines()
#print(sci_list)

calib_sci={}

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
#    cmd_sci2="ls ./ | awk -F'_Astrodon' '{print $1}'| awk '{print substr($0,length-2,3)}' | cut -d - -f2"
    #cmd_sci_filter='echo '+ +'| cut -d / -f4 | awk -F'_Astro' '{print $1}'| rev | cut -c1-3 | rev | cut -d - -f2'
#    sci_filter=os.popen(cmd_sci1,"r").read().splitlines()[0]
#    print(sci_filter)
    #filter_idx=sci_filter
#    array_each_sci=np.array([pyfits.getdata(i)])[0]
#    print(array_each_sci.shape)
    idx_filter_time=sci_filter+"_"+idx_time
#    print(idx_filter_time)
    select_master_flat=master_flat[idx_filter_time]
#    print(select_master_flat[1000][1000])
#    select_master_dark=master_dark[idx_time]
#    print(select_master_dark[1000][1000])
    sci_flat=imdata/select_master_flat
    calib_sci[i]=sci_flat
#    print(time_idx,sci_filter)
#    print(time_idx,sci_filter)
#    print(select_master_flat.shape)
#    print(select_master_dark.shape)
#    print(sci_flat.shape)
    plt.title(i)
    plt.imshow(sci_flat,cmap='rainbow')
    #plt.imshow(imdata)
    plt.show()
    
    


