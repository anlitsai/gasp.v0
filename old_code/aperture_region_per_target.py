#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 05:22:35 2019

@author: altsai
"""

import os
import sys
import numpy as np
import csv
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates

#import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS


'''
#fits_root=input('Please input the file name of the fitsimage: ').split('.',-1)[0].split('_calib',-1)[0]
fits_root='3C345-20190822@130653-R_Astrodon_2018'
fits_calib=fits_root+'_calib.fits'
fits_ori=fits_root+'.fts'
print(fits_root)
print(fits_calib)
print(fits_ori)

date=fits_root.split('@',-1)[0].split('-',-1)[-1]
print(date)
print(type(date))
yearmonth=date[0:6]
year=date[0:4]
month=date[4:6]
'''



#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
file_info='gasp_target_fitsheader_info_slt2019.txt'
#file_info='gasp_target_fitsheader_info_slt'+date+'.txt'
#file_info='gasp_target_fitsheader_info_slt'+year+month+'.txt'
print('... will read '+file_info+' ...')
df_info=pd.read_csv(file_info,delimiter='|')
#print(df_info)


#idx_fitsheader=df_info[df_info['Filename']==fits_ori].index #[0]
#print(idx_fitsheader)
#obj_name=df_info['Object'][idx_fitsheader]
#fwhm=df_info['FWHM'][idx_fitsheader]
#print(fwhm)
#file_sourcelist_name=df_info['Filename'][idx_fitsheader]


'''
ra0_deg=df_info['RA_deg'][idx_fitsheader]
dec0_deg=df_info['DEC_deg'][idx_fitsheader]
'''

'''
idx_fitsheader=df_info[df_info['Object']==obj_name].index 
#fwhm=df_info['FWHM'][idx_fitsheader]
fwhm=max(df_info['FWHM'][idx_fitsheader])
print(fwhm)
'''


obj_name='3C345'
obj_name='3C371'
obj_name='3C454-3'
obj_name='4C38-41'
obj_name='4C51-37'
obj_name='CTA102'
obj_name='DA406'
obj_name='ES2344+514'
obj_name='KS1510-08'
obj_name='L-Lacertae'
obj_name='Mkn501'
obj_name='ON231'
obj_name='PKS2155-304'
obj_name='3C273'
obj_name='3C279'
obj_name='4C29-45'
obj_name='4C71-07'
obj_name='Mkn421'   #leaking from nearby stars
obj_name='S4_0954+65'
obj_name='S5_0716+71'
obj_name='3C66A'
obj_name='AO0235+16'
obj_name='OJ49'
obj_name='OJ248'
obj_name='OJ287'
obj_name='PKS0735+17'


'''
r_circle=fwhm*4.
r_inner=fwhm*7. 
r_outer=fwhm*8. 
'''
r_circle=12.
r_inner=22. 
r_outer=32. 



dir_refstar='RefStar/'
file_refstar='gasp_refStar_radec.txt'

df_refstar=pd.read_csv(file_refstar,sep='|')
#print(df_refstar)
idx_refstar=df_refstar[df_refstar['ObjectName']==obj_name].index.tolist()
n_refstar=len(idx_refstar)


'''
dir_reg=year+'/'+'slt'+date+'_reg/'
if not os.path.exists(dir_reg):
    os.makedirs(dir_reg)
'''
dir_reg=dir_refstar+'annu/'
#dir_reg=dir_refstar
#dir_reg='./'
    
#file_reg_fk5='RefStar_'+obj_name+'_mod_fk5.reg'
#file_reg_fk5=dir_reg+fits_root+'_fk5.reg'
file_reg_fk5=dir_reg+'RefStar_'+obj_name+'_annu_fk5.reg'
print('will write to : '+file_reg_fk5)
if os.path.exists(file_reg_fk5):
    os.remove(file_reg_fk5)
f_reg=open(file_reg_fk5,'w')

f_reg.write('# Region file format: DS9 version 4.1\n')
f_reg.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f_reg.write('fk5\n')

file_reg_ori=dir_refstar+'RefStar_'+obj_name+'_fk5.reg'
cmd_search_head="cat "+file_reg_ori+" | head -3"
line_head=os.popen(cmd_search_head,"r").read().splitlines()
'''
for i in range(3):
    f_reg.write(line_head[i]+'\n')
'''

cmd_search_reg_box="cat "+file_reg_ori+" | grep box"
print(cmd_search_reg_box)
line_box=os.popen(cmd_search_reg_box,"r").read().splitlines()[0]
print(line_box)

cmd_search_text="cat "+file_reg_ori+" | grep circle | grep color | cut -d ')' -f2"
text_color=os.popen(cmd_search_text,"r").read().splitlines()

cmd_search_radec="cat "+file_reg_ori+" | grep circle | cut -d '(' -f2| cut -d ',' -f1,2"
text_radec=os.popen(cmd_search_radec,"r").read().splitlines()


k=0
for i in idx_refstar:
    if k==0:
        txt_target='circle('+str(text_radec[0])+','+str(r_circle)+'")'+text_color[0]
        txt_bkg='annulus('+str(text_radec[0])+','+str(r_inner)+'",'+str(r_outer)+'")'
    else:
        txt_target='circle('+str(text_radec[k])+','+str(r_circle)+'")'+text_color[k]
        txt_bkg='annulus('+str(text_radec[k])+','+str(r_inner)+'",'+str(r_outer)+'")'
#    print(k,i)
    f_reg.write(txt_target+'\n')
    f_reg.write(txt_bkg+'\n')
    k=k+1
    
f_reg.write(line_box+'\n')
f_reg.close()
