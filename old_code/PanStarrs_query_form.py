#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 04:38:37 2019

@author: altsai
"""

import os
import sys
import numpy as np
import csv
import pandas as pd

#n_row=100



print('-------------------------------------')

print('... instrument mag from APT.tbl ...')
file_in_apt='/home/altsai/.AperturePhotometryTool/APT.tbl'
df_apt=pd.read_csv(file_in_apt,delim_whitespace=True,skiprows=2)
#ra_apt=df_apt['ApertureRA']
#dec_apt=df_apt['ApertureDec']
#mag_apt=df_apt['Magnitude']


n_brightest=500
df_apt2=df_apt.sort_values(by='Magnitude',ascending=True)[:n_brightest]
print('number of rows:',len(df_apt2))
#df_apt3=df_apt2[['ApertureRA','ApertureDec','Magnitude']]
#df_apt3=df_apt3.rename(columns={'ApertureRA':'RA','ApertureDec':'Dec','Magnitude':'Instrument Magnitude'})
df_apt3=df_apt2[['ApertureRA','ApertureDec']]
df_apt3=df_apt3.rename(columns={'ApertureRA':'ra','ApertureDec':'dec'})
file_out_apt='PanStarrs_query_radec_'+str(n_brightest)+'.csv'
#df2.to_csv(file_out,sep='\t',index=False,header=True)
#df_apt3.to_csv(file_out_apt,sep='\t',index=False,header=False)
df_apt3.to_csv(file_out_apt,sep=',',index=False,header=True)

print('output file :',file_out_apt)
print('now go to https://catalogs.mast.stsci.edu/panstarrs/ ')


