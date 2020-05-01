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


obj_name='3C345'  # 12,17,22
obj_name='3C371'  # 12,17,22
obj_name='3C454-3'
#obj_name='4C38-41'
#obj_name='4C51-37'
#obj_name='CTA102'
#obj_name='DA406'
#obj_name='ES2344+514'
#obj_name='KS1510-08'
#obj_name='L-Lacertae'
#obj_name='Mkn501'
#obj_name='ON231'
#obj_name='PKS2155-304'
#obj_name='3C273'
#obj_name='3C279'
#obj_name='4C29-45'
#obj_name='4C71-07'
#obj_name='Mkn421'   #leaking from nearby stars
#obj_name='S4_0954+65'
#obj_name='S5_0716+71'
#obj_name='3C66A'
#obj_name='AO0235+16'
#obj_name='OJ49'
#obj_name='OJ248'
#obj_name='OJ287'
#obj_name='PKS0735+17'


'''
r_circle=fwhm*4.
r_inner=fwhm*7. 
r_outer=fwhm*8. 
'''
r_circle=10.
r_inner=15. 
r_outer=20. 



dir_refstar='RefStar/'
file_refstar='gasp_refStar_radec.txt'

df_refstar=pd.read_csv(file_refstar,sep='|')
#print(df_refstar)
idx_refstar=df_refstar[df_refstar['ObjectName']==obj_name].index.tolist()
print(idx_refstar)
n_refstar=len(idx_refstar)


'''
dir_reg_annu=year+'/'+'slt'+date+'_reg/'
if not os.path.exists(dir_reg_annu):
    os.makedirs(dir_reg_annu)
'''
dir_reg_annu=dir_refstar+'annu/'
#dir_reg_annu=dir_refstar
#dir_reg_annu='./'
if not os.path.exists(dir_reg_annu):
    os.makedirs(dir_reg_annu)
    
#file_reg_annu='RefStar_'+obj_name+'_mod_fk5.reg'
#file_reg_annu=dir_reg_annu+fits_root+'_fk5.reg'
file_reg_annu=dir_reg_annu+'RefStar_'+obj_name+'_annu_fk5.reg'
print('will write to : '+file_reg_annu)
if os.path.exists(file_reg_annu):
    os.remove(file_reg_annu)
    

f_reg_ann=open(file_reg_annu,'w')

head1='# Region file format: DS9 version 4.1\n'
head2='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
head3='fk5\n'

f_reg_ann.write(head1)
f_reg_ann.write(head2)
f_reg_ann.write(head3)

#file_reg_ori=dir_refstar+'RefStar_'+obj_name+'_fk5.reg'
file_reg_ori=dir_refstar+'RefStar_'+obj_name+'_bkg_fk5.reg'
#cmd_search_head="cat "+file_reg_ori+" | head -3"
#print(cmd_search_head)
#line_head=os.popen(cmd_search_head,"r").read().splitlines()
#print(line_head)
'''
for i in range(3):
    f_reg_ann.write(line_head[i]+'\n')
'''

cmd_search_reg_box="cat "+file_reg_ori+" | grep box"
#print(cmd_search_reg_box)
row_box=os.popen(cmd_search_reg_box,"r").read().splitlines()[0]
#print(row_box)

cmd_search_obj_text_color="cat "+file_reg_ori+" | grep circle | grep color | cut -d ')' -f2"
#print(cmd_search_obj_text_color)
obj_text_color=os.popen(cmd_search_obj_text_color,"r").read().splitlines()
#print(obj_text_color)


cmd_search_obj_text="cat "+file_reg_ori+" | grep circle | grep color | cut -d ')' -f2 | cut -d ' ' -f4"
#print(cmd_search_obj_text)
obj_text=os.popen(cmd_search_obj_text,"r").read().splitlines()
#print(obj_text)

cmd_search_obj_radec="cat "+file_reg_ori+" | grep circle | grep color |cut -d '(' -f2| cut -d ',' -f1,2"
#print(cmd_search_obj_radec)
obj_radec=os.popen(cmd_search_obj_radec,"r").read().splitlines()
#print(obj_radec)

'''
cmd_search_bkg_radec="cat "+file_reg_ori+" | grep circle | grep color=black |cut -d '(' -f2| cut -d ',' -f1,2"
print(cmd_search_bkg_radec)
bkg_radec=os.popen(cmd_search_bkg_radec,"r").read().splitlines()[0]
print(bkg_radec)

cmd_search_obj_text_black="cat "+file_reg_ori+" | grep circle | grep color=black | cut -d ')' -f2"
#print(cmd_search_obj_text_black)
obj_text_black=os.popen(cmd_search_obj_text_black,"r").read().splitlines()
#print(obj_text_black)
'''


k=0
for i in idx_refstar[:-1]:
#    print(k)
    if k==0:
        txt_target='circle('+str(obj_radec[0])+','+str(r_circle)+'")'+obj_text_color[0]
        txt_bkg='annulus('+str(obj_radec[0])+','+str(r_inner)+'",'+str(r_outer)+'") # '+obj_text[0]
    else:
        txt_target='circle('+str(obj_radec[k])+','+str(r_circle)+'")'+obj_text_color[k]
        txt_bkg='annulus('+str(obj_radec[k])+','+str(r_inner)+'",'+str(r_outer)+'") # '+obj_text[k]
#    print(k,i)
    f_reg_ann.write(txt_target+'\n')
    f_reg_ann.write(txt_bkg+'\n')
    k=k+1
    
f_reg_ann.write(row_box+'\n')
f_reg_ann.close()

print('...write file to :',file_reg_annu)
print()
# ==========================================

dir_reg_circle=dir_refstar+'circle/'
if not os.path.exists(dir_reg_circle):
    os.makedirs(dir_reg_circle)
 
    
file_reg_circle=dir_reg_circle+'RefStar_'+obj_name+'_circle_fk5.reg'
print('will write to : '+file_reg_circle)
if os.path.exists(file_reg_circle):
    os.remove(file_reg_circle)
    
f_reg_circle=open(file_reg_circle,'w')


f_reg_circle.write(head1)
f_reg_circle.write(head2)
f_reg_circle.write(head3)


rr=10
r_bkg=20

k=0
for i in idx_refstar:
#    print(k)
    if k==0:
#        print(k,idx_refstar[k])
        txt_target='circle('+str(obj_radec[0])+','+str(rr)+'")'+obj_text_color[0]
    elif idx_refstar[0] < i < idx_refstar[-1]:
#        print(k,idx_refstar[k])
        txt_target='circle('+str(obj_radec[k])+','+str(rr)+'")'+obj_text_color[k]
#    print(k,i)
    else:
#        print(k,idx_refstar[k])
        txt_target='circle('+str(obj_radec[k])+','+str(r_bkg)+'")'+obj_text_color[k]
    f_reg_circle.write(txt_target+'\n')
#    f_reg_circle.write(txt_bkg+'\n')
    k=k+1
    
f_reg_circle.write(row_box+'\n')
f_reg_circle.close()

print('...write file to :',f_reg_circle)
print()
