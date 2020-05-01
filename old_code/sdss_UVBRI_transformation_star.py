#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:00:20 2019

@author: altsai
"""


import os
import sys
import numpy as np
from astropy.io import ascii
import csv
import pandas as pd

from astroquery.ned import Ned
import astropy.units as unit
from astropy import coordinates
from astroquery.vizier import Vizier
import astropy.coordinates as coord
from astroquery.utils import TableList
#from tabulate import tabulate
from astropy.coordinates import SkyCoord


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


def griz2R(g,r,i):
    if r>-999.0:
        if g>-999.0:
            R = r - 0.1837*(g - r) - 0.0971;  #sigma = 0.0106
            print(R)
        elif i>-999.0:
            R = r - 0.2936*(r - i) - 0.1439;  #sigma = 0.0072
            print(R)
        else:
            R=-999.0
            print('no R')
            
            





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
    if -999.0<r<threshold:
        if -999.0<g<threshold:
            R = r - 0.1837*(g - r) - 0.0971;  #sigma = 0.0106
            k=k+1
            print(R)
        elif -999.0<i<threshold:
            R = r - 0.2936*(r - i) - 0.1439;  #sigma = 0.0072
            k=k+1
            print(R)
        else:
            R=-999.0
            print(R)
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
'''
file_radec='PS-3c345_'+str(int(threshold))+'mag_px.txt'
df3=df2[['raMean','decMean']]
df3=df3.rename(columns={'raMean':'ra','decMean':'dec'})
df3.to_csv(file_radec,sep=',',index=False,header=False)
'''
file_radec='PS-3c345_'+str(int(threshold))+'mag_radecRmag.txt'
df5=df2[['raMean','decMean','Rmag']]
df5=df5.rename(columns={'raMean':'ra','decMean':'dec','Rmag':'Apparent Magnitude'})
df5.to_csv(file_radec,sep='\t',index=False,header=False)
#df5.to_csv(file_radec,sep='\t',index=False,header=True)
