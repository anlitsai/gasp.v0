#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:26:35 2019

@author: altsai
"""
import os
import sys
import numpy as np
#import csv
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import datetime
from astropy.io import fits


print('-------------------------------------')
print('Now confirm the fits data filename you selected')
print('-------------------------------------')




fits_root=input('Please input the file name of the fitsimage: ').split('.',-1)[0].split('_calib',-1)[0]
fits_calib=fits_root+'_calib.fits'
fits_ori=fits_root+'.fts'
#print(fits_root)
#print(fits_calib)
#print(fits_ori)

date=fits_root.split('@',-1)[0].split('-',-1)[-1]
print(date)
print(type(date))

file_info='gasp_target_fitsheader_info_slt'+date+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
print('... will read '+file_info+' ...')
df_info=pd.read_csv(file_info,delimiter='|')
print()

sys.exit(0)



#print(df_info)
#df_info2=df_info[['DateObs','Object','RA_deg','DEC_deg','Filename']]
#obj_name=df_info.loc[df]['Object']

idx_fitsname=df_info[df_info['Filename'].str.contains(fits_root)].index.tolist()[0]
#idx_fitsname=df_info.loc[df_info['Filename']==fitsname].index[0]
#idx_fitsname=df_info2.loc[(df_info2['DateObs']==filter_date) & (df_info2['Object']==obj_name)& (df_info2['Filename']==filename)].index
#id_file=df_info.loc[df_info['Filename']==filename]['ID'].iloc[0]
#print('The ID of your file is: ',id_file)
#print(idx_fitsname)
#print(type(idx_fitsname))

df_obj=df_info.iloc[idx_fitsname:idx_fitsname+1]

#sys.exit(0)


#Nr_obj=int(input("Enter ID of Filename: "))
#idx_obj=int(1)

#obj_name=df_info2['Object'][id_filename]
#obj_ra_hhmmss=df_info2['RA_hhmmss'][idx_filename]
#obj_dec_ddmmss=df_info2['DEC_ddmmss'][idx_filename]
obj_ra_deg=df_info.at[idx_fitsname,'RA_deg']
obj_dec_deg=df_info.at[idx_fitsname,'DEC_deg']
obj_name=df_info.at[idx_fitsname,'Object']
#date_obs=df_info.at[idx_fitsname,'DateObs']
#date=date_obs.replace('-','')
ID=df_info.at[idx_fitsname,'ID']
#obj_dec_deg=df_info.iloc['DEC_deg'][id_file]
#filter_name=df_info2['FilterName'][idx_filename]


ra0_deg=float(obj_ra_deg)
dec0_deg=float(obj_dec_deg)
#ra0_deg=obj_ra_deg
#dec0_deg=obj_dec_deg

#print(ra0_deg,dec0_deg)
#print(type(ra0_deg),type(dec0_deg))



#input("Press Enter to continue...")
#info_choice='Your object is: '+str(obj_name)+' [RA_deg] '+str(obj_ra_deg)+' [DEC_deg] '+str(obj_dec_deg)
#print(info_choice)
#print(type(Nr_obj))


'''
head_info='ID|DateObs|Filename|Object|RA_hhmmss|DEC_ddmmss|RA_deg|DEC_deg|FilterName|JD(day)|ExpTime(sec)|ZMAG(mag)|FWHM|Altitude|Airmass|Rmag'
file_info.write(head_info+'\n')

hdu=fits.open(fits_calib)[0]
imhead=hdu.header
imdata=hdu.data
#print(imdata.shape)
exptime=imhead['EXPTIME']
idx_time=str(int(exptime))+'S'
#print(idx_time)
#print(exptime)
#naxis=imhead['NAXIS']
#print(naxis)
date_obs=imhead['DATE-OBS'].split('T',-1)[0]
#time_obs=imhead['TIME-OBS']
altitude=imhead['ALTITUDE']
airmass=imhead['AIRMASS']
jd=imhead['JD']
obj=imhead['OBJECT']
try: 
    fwhm=imhead['FWHM']
except KeyError:
    fwhm=-9999
try:
    zmag=imhead['ZMAG']
except KeyError:
    zmag=-9999
ra_hhmmss=imhead['RA']
dec_ddmmss=imhead['Dec']
radec_deg=SkyCoord(ra_hhmmss,dec_ddmmss,unit=(u.hourangle,u.deg))
#ra_deg=SkyCoord(ra_hhmmss,unit=(u.hourangle))
#dec_deg=SkyCoord(dec_ddmmss,unit=(u.deg))
ra_deg=radec_deg.ra.deg
dec_deg=radec_deg.dec.deg
#print(ra_deg,dec_deg)
filter_name=imhead['FILTER']
#select_master_dark=master_dark
cmd_sci_filter='echo '+filter_name+'|cut -d _ -f1'
#print(cmd_sci_filter)
sci_filter=os.popen(cmd_sci_filter,"r").read().splitlines()[0]
#print(sci_filter)
idx_filter_time=sci_filter+"_"+idx_time
info_sci=str(k)+' [DATE] '+date_obs+ str(filename_sci)+' [OBJ] '+str(obj)+' [RA_hhmmss] '+ra_hhmmss+' [DEC_ddmmss] '+dec_ddmmss+' [RA_deg] '+str(ra_deg)+' [DEC_deg] '+str(dec_deg)+' [FIL] '+filter_name+' [JD] '+str(jd)+' [EXPTIME] '+str(exptime)+' [ZMAG] '+str(zmag)+' [FWHM] '+str(fwhm)+' [ALT] '+str(altitude)+' [AIRMASS] '+str(airmass)+' [Rmag] '+str(R_mag)
#print(info_sci)
#f_log.write(info_sci+'\n')
info_write=str(idx)+'|'+date_obs+'|'+ str(filename_sci)+'|'+str(obj)+'|'+ra_hhmmss+'|'+dec_ddmmss+'|'+str(ra_deg)+'|'+str(dec_deg)+'|'+filter_name+'|'+str(jd)+'|'+str(exptime)+'|'+str(zmag)+'|'+str(fwhm)+'|'+str(altitude)+'|'+str(airmass)+'|'+str(R_mag)
#f_info.write(info_write+'\n')
    
'''












#input("Press Enter to continue...")
#f.write('---------------\n')
#f.write(info_choice+'\n')
#f.write('---------------\n')





print('-------------------------------------')


print('-------------------------------------')
print('Find the Instrument magnitude of the central object')
print('-------------------------------------')

tbl_catalog='APT.tbl'
#tbl_catalog='APT_'+date+'_'+str(ID)+'.tbl'
#file_in_apt='APT.tbl'
df_apt=pd.read_csv(tbl_catalog,delim_whitespace=True,skiprows=2)[:-1]

df_apt2=df_apt[['ApertureRA','ApertureDec','Magnitude']].astype(float)
n_row=len(df_apt2)
print(n_row)

dxy_deg=np.zeros(n_row)
#print(dxy_deg)
#print(type(dxy_deg))




dx=df_apt2['ApertureRA']-ra0_deg
dy=df_apt2['ApertureDec']-dec0_deg
dxy=np.sqrt(dx**2+dy**2)
df_apt2['dxy_deg']=dxy

#print(df_apt2)
df_apt3=df_apt2.sort_values(by=['dxy_deg']).reset_index(drop=True)
print(df_apt3)

mag0_instrument=df_apt3['Magnitude'][0]
#print(mag0_instrument)

print('------')
#sys.exit(0)

print('-------------------------------------')
print('Now use APT to compute Zeropoint magnitude, bright value back')
print('-------------------------------------')
zeroRmag=float(input('Input zero point Rmag got from APT = '))
#Rmag_target=
print('-------------------------------------')

R_mag=zeroRmag+mag0_instrument
print('Rmag = ',R_mag)

#obj_info['Rmag']=R_mag

file_out='target_Rmag_'+obj_name+'_'+date+'_'+str(idx_fitsname)+'.txt'
if os.path.exists(file_out):
    os.remove(file_out)
f_out=open(file_out,'w')
head_info='ID|DateObs|Filename|Object|RA_hhmmss|DEC_ddmmss|RA_deg|DEC_deg|FilterName|JD|ExpTime_sec|ZMAG|FWHM|Altitude|Airmass|ZeroRmag|InstrumentRmag|Rmag'
#file_info.write(head_info+'\n')

df_obj=df_obj.assign(ZeroRmag=[zeroRmag])
df_obj=df_obj.assign(InstrumentRmag=[mag0_instrument])
df_obj=df_obj.assign(Rmag=[R_mag])

df_obj.to_csv(file_out,sep='|',header=head_info,index=False)


