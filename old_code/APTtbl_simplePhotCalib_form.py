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

print('... instrument magnitude ...')
file_in='APT.tbl'
df=pd.read_csv(file_in,delim_whitespace=True,skiprows=2)
ra=df['ApertureRA']
dec=df['ApertureDec']
mag=df['Magnitude']


file_out='APT_radecmag_ap.txt'
df2=df[['ApertureRA','ApertureDec','Magnitude']]
df2=df2.rename(columns={'ApertureRA':'RA','ApertureDec':'Dec','Magnitude':'Instrument Magnitude'})
#df2.to_csv(file_out,sep='\t',index=False,header=True)
df2.to_csv(file_out,sep='\t',index=False,header=False)

print()

print('... PanStarrs catalog magnitude to Rmag ...')


file_search_Panstarr_13arcm='PS-3c345.csv'
df=pd.read_csv(file_search_Panstarr_13arcm,delimiter=',')

file_search_13arcm='PS-3c345.csv'
df=pd.read_csv(file_search_13arcm,delimiter=',')
ra=df['raMean']
dec=df['decMean']
gmag=df['gMeanPSFMag']
rmag=df['rMeanPSFMag']
imag=df['iMeanPSFMag']
zmag=df['zMeanPSFMag']
#y=df['yMeanPSFMag']
n_row=len(ra)
Rmag=['-999.0']*n_row

k=0
R=-999.0
threshold=16.0
for j in range(n_row):
    r=rmag[j]
    g=gmag[j]
    i=imag[j]
    n=df['nStackDetections'][j]
    if -999.0<r<threshold:
        if -999.0<g<threshold:
            R = r - 0.1837*(g - r) - 0.0971;  #sigma = 0.0106
            k=k+1
            print(n,R)
        elif -999.0<i<threshold:
            R = r - 0.2936*(r - i) - 0.1439;  #sigma = 0.0072
            k=k+1
            print(n,R)
        else:
            R=-999.0
            print(n,R)
    else:R=-999.0
    Rmag[j]=R
    
print('total',k,'/',n_row,'R are converted...')

#print(Rmag)
df['Rmag']=Rmag
print(df['Rmag'])

head0=df.head(0)
print(head0)
print(type(head0))


file_df='PS-3c345_'+str(int(threshold))+'mag.txt'
df2=df[df.Rmag != -999.0]
df2.to_csv(file_df,sep='|',index=True)



file_radec='PS-3c345_'+str(int(threshold))+'mag_radec.txt'
df3=df2[['raMean','decMean']]
df3=df3.rename(columns={'raMean':'ra','decMean':'dec'})
df3.to_csv(file_radec,sep=',',index=False,header=False)
