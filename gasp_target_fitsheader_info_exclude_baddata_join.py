#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  30 15:18:28 2019

@author: altsai
"""

"""
Spyder Editor

"""
import os
import sys
import numpy as np
import pandas as pd

df01=pd.read_csv('gasp_target_fitsheader_info_exclude_baddata_201804-201901.txt',sep='|')
df02=pd.read_csv('gasp_target_fitsheader_info_exclude_baddata_201902-201910.txt',sep='|')
df03=pd.read_csv('gasp_target_fitsheader_info_exclude_baddata_201911-201912.txt',sep='|')
df04=pd.read_csv('gasp_target_fitsheader_info_exclude_baddata_202001.txt',sep='|')
df05=pd.read_csv('gasp_target_fitsheader_info_exclude_baddata_202002.txt',sep='|')
df06=pd.read_csv('gasp_target_fitsheader_info_exclude_baddata_202003.txt',sep='|')
df07=pd.read_csv('gasp_target_fitsheader_info_exclude_baddata_202004.txt',sep='|')

df_all=pd.concat([df01,df02,df03,df04,df05,df06,df07]).reset_index(drop=True)
#print(df_all)
#print(df_all.ID)

#sys.exit(0)
#idx=df_all.values
idx=df_all.index.values
#print(idx)

ID=idx+1
#print(ID)

df_out=df_all
df_out.ID=ID
#print(df_out.ID)

print(df_out)

file_join='gasp_target_fitsheader_info_exclude_baddata_join.txt'

df_out.to_csv(file_join,sep='|',index=False)

print('')
print('... join all table files to '+file_join+' ...')
print('')


