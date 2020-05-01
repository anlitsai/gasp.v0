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

print('... convert PanStarrs catalog magnitude to Rmag ...')

file_in_ps=input('Which PanStarrs target catalog you will use? (ex: PS-3c345.csv) ')
df_ps=pd.read_csv(file_in_ps,delimiter=',')
#ra_ps=df_ps['raMean']
#dec_ps=df_ps['decMean']
#gmag=df_ps['gMeanPSFMag']
#rmag=df_ps['rMeanPSFMag']
#imag=df_ps['iMeanPSFMag']
#zmag=df_ps['zMeanPSFMag']
#y=df['yMeanPSFMag']
n_row=len(df_ps['gMeanPSFMag'])

'''
# http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
ugriz -> BVRI
Lupton (2005)
These equations that Robert Lupton derived by matching DR4 photometry to Peter Stetson's published photometry for stars.
Stars

   B = u - 0.8116*(u - g) + 0.1313;  sigma = 0.0095
   B = g + 0.3130*(g - r) + 0.2271;  sigma = 0.0107

   V = g - 0.2906*(u - g) + 0.0885;  sigma = 0.0129
   V = g - 0.5784*(g - r) - 0.0038;  sigma = 0.0054

   R = r - 0.1837*(g - r) - 0.0971;  sigma = 0.0106
   R = r - 0.2936*(r - i) - 0.1439;  sigma = 0.0072

   I = r - 1.2444*(r - i) - 0.3820;  sigma = 0.0078
   I = i - 0.3780*(i - z)  -0.3974;  sigma = 0.0063
'''

Rmag=['-999.0']*n_row

k=0
R=-999.0
threshold=16.0
for j in range(n_row):
    r=df_ps['rMeanPSFMag'][j]
    g=df_ps['gMeanPSFMag'][j]
    i=df_ps['iMeanPSFMag'][j]
    n=df_ps['nStackDetections'][j]
    if -999.0<r<threshold:
        if -999.0<g<threshold:
            R = r - 0.1837*(g - r) - 0.0971;  #sigma = 0.0106
            k=k+1
#            print(n,R)
        elif -999.0<i<threshold:
            R = r - 0.2936*(r - i) - 0.1439;  #sigma = 0.0072
            k=k+1
#            print(n,R)
        else:
            R=-999.0
#            print(n,R)
    else:R=-999.0
    Rmag[j]=R
    
print('total',k,'/',n_row,'R are converted...')

#print(Rmag)
df_ps['Rmag']=Rmag
#print(df_ps['Rmag'])

head0=df_ps.head(0)
print(head0)
print(type(head0))


df_ps2=df_ps[df_ps.Rmag != -999.0].sort_values(by='Rmag',ascending=True)
df_ps3=df_ps2[['raMean','decMean','Rmag']]
df_ps3=df_ps3.rename(columns={'raMean':'ra','decMean':'dec','Rmag':'Apparent Magnitude'})

file_out_ps='APT_simplePhotCalib_radec_mag_sdss2Rmag.txt'
df_ps3.to_csv(file_out_ps,sep='\t',index=False,header=False)

#sys.exit(0)

print('-------------------------------------')

print('... instrument mag from APT.tbl ...')
file_in_apt='/home/altsai/.AperturePhotometryTool/APT.tbl'
df_apt=pd.read_csv(file_in_apt,delim_whitespace=True,skiprows=2)
#ra_apt=df_apt['ApertureRA']
#dec_apt=df_apt['ApertureDec']
#mag_apt=df_apt['Magnitude']



df_apt2=df_apt.sort_values(by='Magnitude',ascending=True).head(k)
print('number of rows:',len(df_apt2))
df_apt3=df_apt2[['ApertureRA','ApertureDec','Magnitude']]
df_apt3=df_apt3.rename(columns={'ApertureRA':'RA','ApertureDec':'Dec','Magnitude':'Instrument Magnitude'})

file_out_apt='APT_simplePhotCalib_radec_mag_instrument.txt'
#df2.to_csv(file_out,sep='\t',index=False,header=True)
df_apt3.to_csv(file_out_apt,sep='\t',index=False,header=False)



