#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:26:35 2019

@author: altsai
"""
import os
import sys
import numpy as np
import csv
import pandas as pd


print('-------------------------------------')
print('Now confirm the fits data filename you selected')
print('-------------------------------------')


print("Which Date is your data ?")
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
df_info2=df_info[['DateObs','Object','RA_deg','DEC_deg','Filename']]
#print(df_info2)


df_info_date=df_info2.loc[df_info2['DateObs']==filter_date] #.reset_index(drop=True)
df_info_obj=df_info_date.drop_duplicates(subset='Object').reset_index(drop=True)
print(df_info_obj['Object'])
idx_obj=int(input('Enter the index of your target name? '))
print()

obj_name=df_info_obj['Object'][idx_obj]
df_info_fitsname=df_info2.loc[(df_info2['DateObs']==filter_date) & (df_info2['Object']==obj_name)].reset_index(drop=True)
print(df_info_fitsname['Filename'])
idx_fitsname=int(input('Enter the index of your filename? '))
print()

fitsname=df_info_fitsname['Filename'][idx_fitsname]
#print('The ID of your file is;',)

'''
#year='2019'
#file_info='gasp_target_fitsheader_info_slt'+year+'.txt'

date=input('Input the date of your data? (ex: 20190822) ')
file_info='gasp_target_fitsheader_info_slt'+date+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
print('... will read '+file_info+' ...')
df_info=pd.read_csv(file_info,delimiter='|')
print()

fitsname=input('Please input the file name of the fitsimage: ').split('_calib',2)[0]
'''
print(fitsname)





#print(df_info)
#df_info2=df_info[['DateObs','Object','RA_deg','DEC_deg','Filename']]
#obj_name=df_info.loc[df]['Object']

idx_fitsname=df_info[df_info['Filename'].str.contains(fitsname)].index.tolist()[0]
#idx_fitsname=df_info.loc[df_info['Filename']==fitsname].index[0]
#idx_fitsname=df_info2.loc[(df_info2['DateObs']==filter_date) & (df_info2['Object']==obj_name)& (df_info2['Filename']==filename)].index
#id_file=df_info.loc[df_info['Filename']==filename]['ID'].iloc[0]
#print('The ID of your file is: ',id_file)
print(idx_fitsname)
print(type(idx_fitsname))
#sys.exit(0)


#Nr_obj=int(input("Enter ID of Filename: "))
#idx_obj=int(1)

#obj_name=df_info2['Object'][id_filename]
#obj_ra_hhmmss=df_info2['RA_hhmmss'][idx_filename]
#obj_dec_ddmmss=df_info2['DEC_ddmmss'][idx_filename]
obj_ra_deg=df_info.at[idx_fitsname,'RA_deg']
obj_dec_deg=df_info.at[idx_fitsname,'DEC_deg']
obj_name=df_info.at[idx_fitsname,'Object']

#obj_dec_deg=df_info.iloc['DEC_deg'][id_file]
#filter_name=df_info2['FilterName'][idx_filename]


ra0_deg=float(obj_ra_deg)
dec0_deg=float(obj_dec_deg)
#ra0_deg=obj_ra_deg
#dec0_deg=obj_dec_deg

#print(ra0_deg,dec0_deg)
#print(type(ra0_deg),type(dec0_deg))



input("Press Enter to continue...")
info_choice='Your object is: '+str(obj_name)+' [RA_deg] '+str(obj_ra_deg)+' [DEC_deg] '+str(obj_dec_deg)
print(info_choice)
#print(type(Nr_obj))
input("Press Enter to continue...")
#f.write('---------------\n')
#f.write(info_choice+'\n')
#f.write('---------------\n')

print('-------------------------------------')

