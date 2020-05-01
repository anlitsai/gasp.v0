#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 03:54:33 2019

@author: altsai
"""


import os
import sys
#import shutil
#import re
import numpy as np
#import numpy
#from astropy.io import fits
#import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#import glob
#import subprocess
#from scipy import interpolate
#from scipy import stats
#from scipy.interpolate import griddata
#from time import gmtime, strftime
import pandas as pd
#from datetime import datetime
from astropy.io import ascii
#from operator import itemgetter




#folder=input("Enter Folder (ex: slt2019xxxx): ")
#folder='slt20190822'

#dir_month=folder[0:9]
#date_obs=folder[3:11]



print("Which Date you are going to process ?")
date=input("Enter a year-month-date (ex: 20190822): ")
year=date[0:4]
month=date[4:6]
day=date[6:8]

file_info='gasp_target_fitsheader_info_slt'+year+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
print('... will read '+file_info+' ...')
print()

#sys.exit(0)

filter_date=year+'-'+month+'-'+day
print('search info in '+filter_date)

df_info=pd.read_csv(file_info,delimiter='|')

#print(df_info)

#df_info_date=df_info['DateObs']==filter_date

df_info_date=df_info.loc[df_info['DateObs']==filter_date].reset_index(drop=True)

print(df_info_date)

n_date=len(df_info_date.index)
print(n_date)

# sort by Object
#source_sorted=df_info.sort_values('Object', inplace=True)
#print(source_sorted['Object'])

df_obj_radec=df_info_date.drop_duplicates(subset='Object')[['ID','Object','RA_deg','DEC_deg']].reset_index(drop=True)
print(df_obj_radec)




#sys.exit(0)



print("Which File you are going to process ?")

df_info_filename=df_info_date[['ID','Filename']]
print(df_info_filename)



Nr_obj=int(input("Enter ID of Filename: "))
#idx_obj=int(1)

obj_name=df_info['Object'][Nr_obj]
obj_ra_hhmmss=df_info['RA_hhmmss'][Nr_obj]
obj_dec_ddmmss=df_info['DEC_ddmmss'][Nr_obj]
obj_ra_deg=df_info['RA_deg'][Nr_obj]
obj_dec_deg=df_info['DEC_deg'][Nr_obj]
filter_name=df_info['FilterName'][Nr_obj]

input("Press Enter to continue...")
info_choice='Your will work on: '+str(obj_name)+' [RA_deg] '+str(obj_ra_deg)+' [DEC_deg] '+str(obj_dec_deg)
print(info_choice)
print(type(Nr_obj))
input("Press Enter to continue...")
#f.write('---------------\n')
#f.write(info_choice+'\n')
#f.write('---------------\n')


#sys.exit(0)


#tbl_catalog=input("Enter file name (APT.tbl): ")
tbl_catalog='APT.tbl'


ra0_deg=obj_ra_deg.tolist()
dec0_deg=obj_dec_deg.tolist()
print(ra0_deg,dec0_deg)
print(type(ra0_deg),type(dec0_deg))
#input("Press Enter to continue...")


#sys.exit(0)

df=pd.read_csv(tbl_catalog,header=2,delim_whitespace=True)[:-1]
print(df)


    
#print(list(df.columns))
#print(type(df.columns))
#print(list(df.columns.values.tolist()))
#print(data['ID'])
df2=df[['Number','ApertureRA','ApertureDec','SourceIntensity','Magnitude']].astype(float)
#df2=df[['ID','ApertureRA','ApertureDec','ApertureX','ApertureY','SourceIntensity','Magnitude','MagUncertainty']].astype(float)
#df2=data[['SourceIntensity','DataUnits','Magnitude']].copy()
print(df2)

df3=df[['Number','ApertureRA','ApertureDec']].astype(float)


mag=df2['Magnitude']
#mag_uncert=df2['MagUncertainty']
ra_aper_deg=df2['ApertureRA']
dec_aper_deg=df2['ApertureDec']

dx=df2['ApertureRA']-ra0_deg
dy=df2['ApertureDec']-dec0_deg
dxy=np.sqrt(dx**2+dy**2)

df2['dxy_deg']=dxy
print(df2)
df2_sort=df2.sort_values(by=['dxy_deg'])
print(df2_sort)
print('------')
#print('df2_sort.head(1)')
#print(df2_sort.head(1))
#print('------')
#print('df2_sort.head(1).dxy_deg')
#print(df2_sort.head(1).dxy_deg)
#print('------')
#print('df2_sort.head(1).dxy_deg.tolist()')
#print(df2_sort.head(1).dxy_deg.tolist())
#print('------')
#print("df2_sort.head(1)['ID'].tolist()")
#print(df2_sort.head(1)['ID'].tolist())
#print('------')
#print('df2_sort.iloc[0,:]')
#print(df2_sort.iloc[0,:])
#print('------')
#print("df2_sort.iloc[0,:]['dxy_deg']")
#print(df2_sort.iloc[0,:]['dxy_deg'])
#print('------')
#print("df2_sort.iloc[0,:]['Magnitude']")
#print(df2_sort.iloc[0,:]['Magnitude'])
#print('------')
#print("df2_sort.iloc[0,:]['ID']")
#print(df2_sort.iloc[0,:]['ID'])
print('------')
print('... search for the n-th row of the source center ...')
print("idx_center=int(df2_sort.iloc[0,:]['Number']-1)")
idx_center=int(df2_sort.iloc[0,:]['Number']-1)
print('... the source center is in row Nr.',idx_center)
input("Press Enter to continue...")
print("df.iloc[idx_center,:]")
print(df.iloc[idx_center,:])
print('------')

