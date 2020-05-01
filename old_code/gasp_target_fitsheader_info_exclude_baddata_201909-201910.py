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
import pandas as pd
from astropy.wcs import WCS
from datetime import datetime
#from scipy import interpolate
#from scipy import stats
from astropy import units as u
from astropy.coordinates import SkyCoord
# https://docs.astropy.org/en/stable/coordinates/

print('-----------------')
print('input parameters')
print('-----------------')

'''
print('... will generate table from fitsheader from 201902')
till_month=str(input('till which month you want to generate table (ex: 201909)? '))
print('... will grap target information till '+till_month+' ...')


year=input("Which Year you are going to process (ex: 2019)? ")
#folder=sys.argv[1]
#folder='slt20190822'
dir_year='slt'+year

till_month=till_month[0:4]+'-'+till_month[4:6]+'-'
print(till_month)
'''
#date=folder[3:11]
#print(dir_month)
#dir_master=dir_month+'_master/'
#print(dir_master)
#dir_calib_sci=folder+'_calib_sci/'
#print(dir_calib_sci)

file_info='gasp_target_fitsheader_info_exclude_baddata_201909-201910.txt'
if os.path.exists(file_info):
    os.remove(file_info)
f_info=open(file_info,'w')

file_log='gasp_target_fitsheader_info_exclude_baddata_201909-201910.log'
if os.path.exists(file_log):
    os.remove(file_log)
f_log=open(file_log,'w')

#time_calib_start=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#time_calib_start=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
time_calib_start=str(datetime.now())  
info_time=('File generated by An-Li Tsai at '+time_calib_start+' UTC')
print(info_time)
f_log.write(info_time+'\n')

print(sys.argv)
f_log.write(str(print(sys.argv))+'\n')

#cmd_search_folders='find ./2019?? |grep calib_sci|grep fits|cut -d / -f3| sort | uniq'
#cmd_search_folders='find ./|grep '+dir_year+'|cut -d / -f3|cut -c-11|sort| uniq|grep -E "'+dir_year+'[0-9][0-9][0-9][0-9]"'
#print(cmd_search_folders)
#f_log.write(cmd_search_folders+'\n')
#list_folders=os.popen(cmd_search_folders,"r").read().splitlines()

cmd_search_folder1='find ./|grep calib | grep slt201909| cut -d / -f3 | sort |uniq'
f_log.write(cmd_search_folder1+'\n')
list_folder1=os.popen(cmd_search_folder1,"r").read().splitlines()
cmd_search_folder2='find ./|grep calib | grep slt201910| cut -d / -f3 | sort |uniq'
f_log.write(cmd_search_folder2+'\n')
list_folder2=os.popen(cmd_search_folder2,"r").read().splitlines()
list_folders=list_folder1+list_folder2
print(list_folders)
info_folders='Your folders are : '+str(list_folders)
#print(info_folders)
f_log.write(info_folders+'\n')


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







print(' ---------------------------')
print(' Science Target ')
print(' ---------------------------')
f_log.write(' ---------------------------\n')
f_log.write(' Science Target \n')
f_log.write(' ---------------------------\n')

'''
cmd_search_n_file_sci="find ./20???? |grep 'fits\|fts'|grep GASP |sort |wc -l"

cmd_search_n_row_till_month='cat '+file_info+'|grep '+till_month+' |tail -1 |cut -d "|" -f1'
print(cmd_search_n_row_till_month)
n_row_till_month=int(os.popen(cmd_search_n_row_till_month,"r").read().splitlines()[0])+1
print(n_row_till_month)
'''
'''
cmd_keep_row_till_month='cat '+file_info+' |head -'+str(n_row_till_month)
print(cmd_keep_row_till_month)
n_row_all=os.popen(cmd_search_n_row_all,"r").read().splitlines()
print(n_row_all)
n_row_need=n_row_all-n_row_till_month
print(n_row_need)
'''







cmd_search_file_sci1="find ./201909 |grep 'fits\|fts'|grep calib_sci |sort "
print(cmd_search_file_sci1)
f_log.write(cmd_search_file_sci1+'\n')
list_file_sci1=os.popen(cmd_search_file_sci1,"r").read().splitlines()
cmd_search_file_sci2="find ./201910 |grep 'fits\|fts'|grep calib_sci |sort "
print(cmd_search_file_sci2)
f_log.write(cmd_search_file_sci2+'\n')
list_file_sci2=os.popen(cmd_search_file_sci2,"r").read().splitlines()
list_file_sci=list_file_sci1+list_file_sci2
#cmd_search_file_sci="find ./20???? |grep 'fits\|fts'|grep calib_sci |sort "
#print(cmd_search_file_sci)
#f_log.write(cmd_search_file_sci+'\n')
#list_file_sci=os.popen(cmd_search_file_sci,"r").read().splitlines()
#info_print='...calibrating science targets...'
#print(info_print)
#f_log.write(info_print+'\n')
print(list_file_sci)
f_log.write(str(list_file_sci)+'\n')
n_file_sci=len(list_file_sci)
info_n_sci='... found '+str(n_file_sci)+' science targets ...'
#print(info_n_sci)
f_log.write(info_n_sci+'\n')




#cmd_search_sci_filter="find ./ |grep "+date"|grep fts |grep GASP|cut -d / -f4|awk -F'_Astro' '{print $1}'| rev|cut -c1-3|rev|cut -d - -f2 "

#os.chdir(dir_root+"wchen/wchen_03_GASP_01/")

#cmd_sci1="ls ./|awk -F'_Astrodon' '{print $1}'| awk '{print substr($0,length-2,3)}'|cut -d - -f2 |sort |uniq"
#print(sci_filter_list)

#sci_list=os.popen("ls","r").read().splitlines()
#print(sci_list)

#calib_sci={}

'''
cmd_search_baddata="cat ./bad_data_note.txt|cut -f1 "
print(cmd_search_baddata)
f_log.write(cmd_search_baddata+'\n')
list_baddata=os.popen(cmd_search_baddata,"r").read().splitlines()
print(list_baddata)
n_baddata=len(list_baddata)
print(n_baddata)
f_log.write(str(list_baddata)+'\n')
'''

#df_baddata_note=pd.read_csv('bad_data_note.txt',sep='\t',skiprows=[0])
df_baddata_note=pd.read_csv('bad_data_note.txt',sep='\t')
print(df_baddata_note)

df_baddata=df_baddata_note.loc[(df_baddata_note['NoteIdx']==1)].reset_index(drop=True)
n_baddata=len(df_baddata)
print(n_baddata)
f_log.write('... total '+str(n_baddata)+' files ... \n')

list_baddata=df_baddata['filename'].tolist()
print(list_baddata)
f_log.write(str(list_baddata)+'\n')



head_info='ID|DateObs|TimeObs|Filename|Object|RA_hhmmss|DEC_ddmmss|RA_deg|DEC_deg|RA_pix|Dec_pix|FilterName|JD|ExpTime_sec|Zmag|FWHM|Altitude|Airmass'
f_info.write(head_info+'\n')

k=0
n_baddata=0
for i in list_file_sci:
    filename_sci=[i.split('/',-1)[-1]][0]
    file_ori=filename_sci.split('_calib')[0]+'.fts'
    hdu=fits.open(i)[0]
    imhead=hdu.header
    imdata=hdu.data
#    print(imdata.shape)
    if list_baddata.count(file_ori)==0:  
#        print(file_ori)
        k=k+1
        idx=str(k)
        print('... process',idx,'/',n_file_sci, '...',i)    
        ra_deg=imhead['CRVAL1']
        exptime=imhead['EXPTIME']
        idx_time=str(int(exptime))+'S'
#        print(idx_time)
#        print(exptime)
#        naxis=imhead['NAXIS']
#        print(naxis)
#        date_obs=imhead['DATE-OBS']
        date_obs=imhead['DATE-OBS'].split('T',-1)[0]
        time_obs=imhead['TIME-OBS'].split('T',-1)[0]
        altitude=imhead['ALTITUDE']
        airmass=imhead['AIRMASS']
        jd=imhead['JD']
        filter_name=imhead['FILTER']
        obj=imhead['OBJECT']
        try: 
            fwhm=imhead['FWHM']
        except KeyError:
            fwhm=-9999
        try:
            zmag=imhead['ZMAG']
        except KeyError:
            zmag=-9999
#        imhead['Rmag']=-9999
#        imhead['Vmag']=-9999
        ra_hhmmss=imhead['RA']
        dec_ddmmss=imhead['Dec']
#        ra_deg=imhead['CRVAL1']
#        dec_deg=imhead['CRVAL2']
#        coordinate convertor
#        https://docs.astropy.org/en/stable/coordinates/
        radec_deg=SkyCoord(ra_hhmmss,dec_ddmmss,unit=(u.hourangle,u.deg))
#        ra_deg=SkyCoord(ra_hhmmss,unit=(u.hourangle))
#        dec_deg=SkyCoord(dec_ddmmss,unit=(u.deg))
        ra_deg=radec_deg.ra.deg
        dec_deg=radec_deg.dec.deg
#        print(ra_deg,dec_deg)
#        select_master_dark=master_dark
        wcs=WCS(imhead)
        xdec_pix=wcs.all_world2pix(ra_deg,dec_deg,1)
        ra_pix=xdec_pix[0].tolist()
        dec_pix=xdec_pix[1].tolist()
#        print(ra_pix,dec_pix)
        cmd_sci_filter='echo '+filter_name+'|cut -d _ -f1'
#        print(cmd_sci_filter)
        sci_filter=os.popen(cmd_sci_filter,"r").read().splitlines()[0]
#        print(sci_filter)
        idx_filter_time=sci_filter+"_"+idx_time
        info_sci=str(k)+' [DATE] '+date_obs+ ' [TIME] '+time_obs+' [FILE] '+str(filename_sci)+' [OBJ] '+str(obj)+' [RA_hhmmss] '+ra_hhmmss+' [DEC_ddmmss] '+dec_ddmmss+' [RA_deg] '+str(ra_deg)+' [DEC_deg] '+str(dec_deg)+' [ra_pix] '+str(ra_pix)+' [dec_pix] '+str(dec_pix)+' [FIL] '+filter_name+' [JD] '+str(jd)+' [EXPTIME] '+str(exptime)+' [ZMAG] '+str(zmag)+' [FWHM] '+str(fwhm)+' [ALT] '+str(altitude)+' [AIRMASS] '+str(airmass)
#        print(info_sci)
        info_write=str(idx)+'|'+date_obs+'|'+time_obs+'|'+ str(filename_sci)+'|'+str(obj)+'|'+ra_hhmmss+'|'+dec_ddmmss+'|'+str('%.4f' %ra_deg)+'|'+str('%.4f' %dec_deg)+'|'+str('%.4f' %ra_pix)+'|'+str('%.4f' %dec_pix)+'|'+filter_name+'|'+str(jd)+'|'+str(exptime)+'|'+str('%.4f' %zmag)+'|'+str('%.4f' %fwhm)+'|'+str('%.4f' %altitude)+'|'+str('%.4f' %airmass)
        f_log.write(info_sci+'\n')
        f_info.write(info_write+'\n')
    else:
        n_baddata=n_baddata+1
#        print(n_baddata)
        f_log.write(filename_sci+' is bad data ...\n')


print('... there are '+str(n_baddata)+' files bad data ...\n')    

f_log.write('\n')    
f_log.write('... there are '+str(n_baddata)+' files bad data ...\n')    
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
#    cmd_sci_name='echo '+i+'|cut -d / -f5|cut -d . -f1'
#    print(cmd_sci_name)
#    sci_name=os.popen(cmd_sci_name,"r").read().splitlines()[0]
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

print()
info1='... write header information to '+file_info+' ... '
print(info1)
f_log.write(info1+'\n')
info2='... finished ...'
print(info2)
f_log.write(info2+'\n')


f_info.close()
f_log.close()
    
