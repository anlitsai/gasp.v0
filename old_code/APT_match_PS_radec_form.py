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
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.table import Table

#n_ps=100


print('-------------------------------------')

print('... convert PanStarrs catalog magnitude to Rmag ...')

file_in_ps=input('Which PanStarrs target catalog you will use? (ex: PS-3c345.csv) ')
df_ps=pd.read_csv(file_in_ps,delimiter=',')
ra_ps=df_ps['raMean']
dec_ps=df_ps['decMean']
#gmag=df_ps['gMeanPSFMag']
#rmag=df_ps['rMeanPSFMag']
#imag=df_ps['iMeanPSFMag']
#zmag=df_ps['zMeanPSFMag']
#y=df['yMeanPSFMag']
n_ps=len(df_ps['gMeanPSFMag'])
print('... PS-xxx.csv has',n_ps,'rows ...')
input('Press Enter to continue...')

idx_ps=np.arange(0,n_ps,1)





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

Rmag=['-999.0']*n_ps

k=0
R=-999.0
threshold=16.0
for j in range(n_ps):
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
    
print('total',k,'/',n_ps,'R are converted...')

#print(Rmag)
df_ps['Rmag']=Rmag
#print(df_ps['Rmag'])

name_column=df_ps.columns
#print(name_column)
#print(type(name_column))


#df_ps_bright=df_ps[df_ps.Rmag != -999.0].sort_values(by='Rmag',ascending=True)
df_ps_bright=df_ps[df_ps.Rmag > -999.0]
ra_ps_bright=df_ps_bright['raMean'] #.reset_index(drop=True)
dec_ps_bright=df_ps_bright['decMean'] #.reset_index(drop=True)
Rmag_ps_bright=df_ps_bright['Rmag'] #.reset_index(drop=True)

n_ps_bright=len(df_ps_bright)
#df_ps_bright_sort=df_ps_bright.sort_values(by='Rmag',ascending=True)
#df_ps3=df_ps_bright_sort[['raMean','decMean','Rmag']]
#df_ps3=df_ps3.rename(columns={'raMean':'ra','decMean':'dec','Rmag':'Apparent Magnitude'})

#file_out_ps='APT_simplePhotCalib_radec_mag_sdss2Rmag.txt'
#df_ps3.to_csv(file_out_ps,sep='\t',index=False,header=False)



coord_ps_bright=SkyCoord(ra=ra_ps_bright*u.degree,dec=dec_ps_bright*u.degree)

#idx_ps,d2d,d3d=coord_apt.match_to_catalog_sky(coord_ps)






#sys.exit(0)

print('-------------------------------------')



#file_in_apt='/home/altsai/.AperturePhotometryTool/APT.tbl'
print('... instrument mag from APT.tbl ...')
file_in_apt='APT.tbl'
df_apt=pd.read_csv(file_in_apt,delim_whitespace=True,skiprows=2)[:-1]
#df_apt=df_apt.convert_objects(convert_numeric=Tru
ra_apt=df_apt['ApertureRA'].astype(float)
dec_apt=df_apt['ApertureDec'].astype(float)
mag_apt=df_apt['Magnitude'].astype(float)
n_apt=len(mag_apt)
print('... APT.tbl has',n_apt,'rows ...')
input('Press Enter to continue...')

coord_apt=SkyCoord(ra=ra_apt*u.degree,dec=dec_apt*u.degree)

#df_apt2=df_apt.sort_values(by='Magnitude',ascending=True) #.head(k)
#print('number of rows:',len(df_apt2))
#df_apt3=df_apt2[['ApertureRA','ApertureDec','Magnitude']]
#df_apt3=df_apt3.rename(columns={'ApertureRA':'RA','ApertureDec':'Dec','Magnitude':'Instrument Magnitude'})

#file_out_apt='APT_simplePhotCalib_radec_mag_instrument.txt'
#df2.to_csv(file_out,sep='\t',index=False,header=True)
#df_apt3.to_csv(file_out_apt,sep='\t',index=False,header=False)



#idx_ps,sep,d1=coord_apt.match_to_catalog_3d(coord_ps_bright)
#matches=coord_ps[idx_ps]

#idx_ps,sep,d1=match_coordinates_sky(coord_apt,coord_ps_bright)
#idx_ps,sep,d1=match_coordinates_sky(coord_apt.frame,coord_ps.frame)
print('-------------------------------------')
print('... match two catalogs of RA Dec ...')

max_sep=1.0*u.arcsec
#idx_ps,sep,d1=match_coordinates_sky(coord_apt,coord_ps)
idx_apt,sepa,d1=coord_ps_bright.match_to_catalog_3d(coord_apt)
sep_constraint=sepa < max_sep


#print(sep_constraint)
n_match=sum(sep_constraint)
print('... found',n_match,'sources match')
idx_apt_match=idx_apt[sep_constraint]
#print(idx_apt_match)
coord_apt_match=coord_apt[idx_apt_match]
#print(idx_apt_match)
#print(len(idx_apt_match))
idx_ps_bright_match=np.arange(0,n_ps_bright,1)[sep_constraint]
#print(idx_ps_bright_match)
coord_ps_bright_match=coord_ps_bright[sep_constraint]
# or coord_ps_bright_match=coord_ps_bright[idx_ps_bright_match]




input('Press Enter to continue...')



ra_ps_match=coord_ps_bright_match.ra.deg
dec_ps_match=coord_ps_bright_match.dec.deg
#Rmag_ps_match=Rmag_ps_bright[idx_apt[sep_constraint]].tolist()
Rmag_ps_match=Rmag_ps_bright.reset_index(drop=True)[idx_ps_bright_match].tolist()

#sys.exit(0)

ra_apt_match=coord_apt_match.ra.deg
dec_apt_match=coord_apt_match.dec.deg
mag_apt_match=mag_apt[idx_apt_match].tolist()
#table_ps_match=Table([ra_ps_match,dec_ps_match],names=('ra','dec'))
#table_apt_match=Table([ra_apt_match,dec_apt_match],names=('ra','dec'))
#df_ps_from_array=pd.DataFrame.from_records(table_ps_match.as_array())
#df_ps_from_table_method=table_ps_match..to_pandas()


#sys.exit(0)




file_out_ps='APT_match_PS_radec_mag_sdss2Rmag.txt'
if os.path.exists(file_out_ps):
    os.remove(file_out_ps)

file_out_apt='APT_match_PS_radec_mag_instrument.txt'
if os.path.exists(file_out_apt):
    os.remove(file_out_apt)

#sys.exit(0)


f_ps=open(file_out_ps,'w')
#f_ps.write('ra,dec,\n')
for i in range(n_match):
    f_ps.write(str(ra_ps_match[i])+'\t'+str(dec_ps_match[i])+'\t'+str(Rmag_ps_match[i])+'\n')
f_ps.close()

#sys.exit(0)

f_apt=open(file_out_apt,'w')
#f_apt.write('ra,dec\n')
for i in range(n_match):
    f_apt.write(str(ra_apt_match[i])+'\t'+str(dec_apt_match[i])+'\t'+str(mag_apt_match[i])+'\n')
f_apt.close()

print()
print('generate 2 files:')
print(file_out_ps)
print(file_out_apt)
print()
print('... now go to "APT Simple Photometry Calibrator" and feed these 2 files ...')
print()
print('... finished ...')