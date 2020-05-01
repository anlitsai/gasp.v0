#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:48:35 2019

@author: altsai
"""

import os
import sys
import numpy as np
from astropy.io import ascii
import csv
import pandas as pd

from astroquery.ned import Ned
import astropy.units as u
from astropy import coordinates
from astroquery.vizier import Vizier
import astropy.coordinates as coord
from astroquery.utils import TableList
#from tabulate import tabulate
from astropy.coordinates import SkyCoord

print('---------------------------------')

file_cata_Landolt='catalog_standard_star_Landolt.txt'

df_cata_Landolt=pd.read_csv(file_cata_Landolt,delimiter='\t')
#print(df_cata_Landolt)
star_name=df_cata_Landolt['Star']
ra0_deg=df_cata_Landolt['RA_deg']
dec0_deg=df_cata_Landolt['DEC_deg']
ra0_hhmmss=df_cata_Landolt['RA_hhmmss']
dec0_ddmmss=df_cata_Landolt['DEC_ddmmss']

#sys.exit(0)
print('---------------------------------')
#input("Press Enter to continue...")
print()
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
#sys.exit(0)

#df_obj_radec=df_info_date.drop_duplicates(subset='Object')[['ID','Object','RA_deg','DEC_deg']].reset_index(drop=True)
df_obj_radec=df_info_date.drop_duplicates(subset='Object')[['Object','RA_deg','DEC_deg','RA_hhmmss','DEC_ddmmss']].reset_index(drop=True)
print('... source table ...')
print(df_obj_radec)
#sys.exit(0)
idx_source=int(input("Enter the index of the source: "))
ra2_deg=df_obj_radec['RA_deg'][idx_source]
dec2_deg=df_obj_radec['DEC_deg'][idx_source]
ra2_hhmmss=df_obj_radec['RA_hhmmss'][idx_source]
dec2_ddmmss=df_obj_radec['DEC_ddmmss'][idx_source]
obj_name=df_obj_radec['Object'][idx_source]
#print('[OBJ]',obj_name,'[RA]',ra0_deg,' [DEC]',dec0_deg,'[RA]',ra2_hhmmss,' [DEC]',dec2_ddmmss)
print('[OBJ]',obj_name,'[RA]',ra2_hhmmss,' [DEC]',dec2_ddmmss)



dx=abs(ra2_deg-ra0_deg)
dy=abs(dec2_deg-dec0_deg)
dxy=np.sqrt(dx**2+dy**2)
df_cata_Landolt['dxy_deg']=dxy

df_cata_sort=df_cata_Landolt.sort_values(by=['dxy_deg']).reset_index(drop=True)

print('[OBJ]',df_cata_sort['Star'][0],'[RA]',df_cata_sort['RA_deg'][0],'[DEC]',df_cata_sort['DEC_deg'][0],'[dxy]',df_cata_sort['dxy_deg'][0])

print()
print('... the most closest 10 stars ...')
print()
print(df_cata_sort.head(10))

file_cata='search_standard_star_Landolt_'+obj_name+'_sort_distance.txt'
df_cata_sort.to_csv(file_cata,sep='|',index=True)