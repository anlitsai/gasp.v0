#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:54:15 2019

@author: altsai
"""
# https://astroquery.readthedocs.io/en/latest/ned/ned.html
import os
import sys
import numpy as np
from astropy.io import ascii
import csv
import pandas as pd

from astroquery.ned import Ned
import astropy.units as u
from astropy import coordinates
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = 100000000000
import astropy.coordinates as coord
from astroquery.utils import TableList
#from tabulate import tabulate
from astropy.coordinates import SkyCoord


print("Which Date you are going to process ?")
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

#df_info_date=df_info['DateObs']==filter_date

df_info_date=df_info.loc[df_info['DateObs']==filter_date].reset_index(drop=True)

print(df_info_date)

n_date=len(df_info_date.index)
print(n_date)


#df_obj_radec=df_info_date.drop_duplicates(subset='Object')[['ID','Object','RA_deg','DEC_deg']].reset_index(drop=True)
df_obj_radec=df_info_date.drop_duplicates(subset='Object')[['Object','RA_deg','DEC_deg','RA_hhmmss','DEC_ddmmss']].reset_index(drop=True)
print('... source table ...')
print(df_obj_radec)

idx_source=int(input("Enter the index of the source: "))
ra0_deg=df_obj_radec['RA_deg'][idx_source]
dec0_deg=df_obj_radec['DEC_deg'][idx_source]
ra_hhmmss=df_obj_radec['RA_hhmmss'][idx_source]
dec_ddmmss=df_obj_radec['DEC_ddmmss'][idx_source]
obj_name=df_obj_radec['Object'][idx_source]
print('[OBJ]',obj_name,'[RA]',ra0_deg,' [DEC]',dec0_deg,'[RA]',ra_hhmmss,' [DEC]',dec_ddmmss)

#sys.exit(0)

r1_deg=0.001
r1_as=r1_deg*3600
print('... will query the data archive with radius =',r1_as,'[as] ...')
input("Press Enter to continue...")

result = Vizier.query_region(coord.SkyCoord
                             (ra=ra0_deg, dec=dec0_deg,unit=(u.deg, u.deg),
                              frame='icrs'),radius=r1_deg*u.deg) 

'''
result = Vizier.query_region(coord.SkyCoord
                             (ra=ra0_deg, dec=dec0_deg,unit=(u.deg, u.deg),
                              frame='icrs'),width=90, height=360,catalog='II/183A/table2') 


#Landlot: II/183A/table2
#UCAC4: I/322A
'''

print(result)
#print(result[-1][0])

n_catalog=len(result)

k=0


#for i in range(n_catalog):
#    try:
#        Rmag_yes=result[i]['Rmag'].tolist()[0]
#        header=result[i].colnames
#        print(result.keys()[i])
#        print(header)
#    except:
#        Rmag=0


#sys.exit(0)

'''
list_cata=[]
for i in range(n_catalog):
    try:
        Rmag_yes=result[i]['Rmag'].tolist()[0]
#        ii=i+2
        cata=result.keys()[i]
        list_cata.append(str(i)+':'+cata)
        info=str(i)+':'+cata+' | Rmag = '+str(Rmag_yes)
        print(info)
        k=k+1



    except:
        Rmag_no='no'

print(list_cata)

sys.exit(0)
'''

list_cata=[]
for i in range(n_catalog):
    try:
        Vmag_yes=result[i]['Vmag'].tolist()[0]
        try:
            Rmag_yes=result[i]['Rmag'].tolist()[0]
            ii=i+2
            cata=result.keys()[ii]
            list_cata.append(str(ii)+':'+cata)
            info=str(ii)+':'+cata+' | Vmag = '+str(Vmag_yes)+' | Rmag = '+str(Rmag_yes)
            print(info)
            k=k+1
        except:
            try:
                VRmag_yes=result[i]['V-Rmag'].tolist()[0]
                ii=i+2
                cata=result.keys()[ii]
                list_cata.append(str(ii)+':'+cata)
                info=str(ii)+':'+cata+' | V-Rmag = '+str(VRmag_yes)
                print(info)
                k=k+1
            except:
                Vmag_no='no'
    except:
#    except KeyError:
        Vmag_no='no'
      
        

print()
info1='... source found within radius =  '+str(r1_as)+' arcsec ...'
print(info1)
info1='... found Vmag Rmag in '+str(k)+' of total '+str(n_catalog)+' catalogs ...'
print(info1)



print('... list of catalog ...')
print(list_cata)

print('---------')
#sys.exit(0)



'''
result_table = Ned.query_object(obj)
print(result_table)

result_table = Ned.query_region(obj,radius=0.01*u.deg)
print(result_table)



co = coordinates.SkyCoord(ra=ra1, dec=dec1, unit=(u.deg, u.deg), frame='fk5')
result_table = Ned.query_region(co, radius=0.01 * u.deg, equinox='J2000.0')
print(result_table)


result_table = Ned.get_table(obj, table='positions')
print(result_table)


result_table = Ned.get_table(obj, table='photometry')
print(result_table)
'''



#v = Vizier(columns=['_RAJ2000', '_DEJ2000','Vmag','Rmag', 'Imag'],column_filters={"Vmag":">10"}, keywords=["optical"])
#result = v.query_object(obj,catalog='I/305/out')
#print(result)

#v = Vizier(catalog="I/305/out",columns=['*','_RAJ2000','_DEJ2000']).query_constraints(Vmag="10.0..11.0")[0]
#print(v)


#agn = Vizier(catalog="VII/258/vv10", columns=['*', '_RAJ2000', '_DEJ2000']).query_constraints(Vmag="10.0..11.0")[0]
#print(agn)
#guide = Vizier(catalog="II/246", column_filters={"Kmag":"<9.0"}).query_region(agn, radius="30s", inner_radius="2s")[0]
#guide.pprint()


#catalog_list = Vizier.find_catalogs('Kang W51')
#catalogs = Vizier.get_catalogs(catalog_list.keys())
#print(catalogs)

print()
r2_arcm=float(input("Enter a query size with radius in arcm (ex: 10 for 10arcm): "))
r2_deg=r2_arcm/60.
#r2_deg=0.2
#r2_arcm=r2_deg*60
print('... will query the data archive with radius =',r2_arcm,'[arcm] ...')
input("Press Enter to continue...")


result2 = Vizier.query_region(coord.SkyCoord
                             (ra=ra0_deg, dec=dec0_deg,unit=(u.deg, u.deg),
                              frame='icrs'),radius=r2_deg*u.deg,
                              catalog=["NOMAD", "UCAC","SDSS","LQRF","GSC"])
                              
print(result2)

n_catalog2=len(result2)


k2=0

list_cata2=[]
'''
print("Which band are you goin to search?")
idx_mag=int(input("Enter (1) Vmag, (2) R_mag, (3) both: "))
'''
info2='... source found within radius =  '+str(r2_arcm)+' arcmin ...'
print(info2)

'''
if idx_mag == 1:
    search_mag='Vmag'
    for i in range(n_catalog2):
        try:
            Rmag_yes=result2[i]['Vmag'].tolist()[0]
            ii=i+2
            cata2=result2.keys()[ii]
            list_cata2.append(str(ii)+':'+cata2)
            k2=k2+1
        except:
            Vmag_no='no'
    info2='... found '+search_mag+' in '+str(k2)+' of total '+str(n_catalog2)+' catalogs ...'
    print(info2)
elif idx_mag == 2:
    search_mag='Rmag'
    for i in range(n_catalog2):
        try:
            Vmag_yes=result2[i]['Rmag'].tolist()[0]
            ii=i+2
            cata2=result2.keys()[ii]
            list_cata2.append(str(ii)+':'+cata2)
            k2=k2+1
        except:
            Rmag_no='no'
    info2='... found '+search_mag+' in '+str(k2)+' of total '+str(n_catalog2)+' catalogs ...'
    print(info2)
else:
    RVmag_no='no'
'''

for i in range(n_catalog2):
    try:
        Rmag_yes=result2[i]['Rmag'].tolist()[0]
#        ii=i
        cata2=result2.keys()[i]
        list_cata2.append(str(i)+':'+cata2)
        k2=k2+1
    except:
        Rmag_no='no'

print()
info2='... found Rmag in '+str(k2)+' of total '+str(n_catalog2)+' catalogs ...'
print(info2)
print(list_cata2)
print()

#sys.exit(0)

print('Which catalog you are going to search?')
#cata_name=[' ']*k2
for i in range(k2):
    cata_info=list_cata2[i].split(':',-1)
#    cata_idx=cata_info[0]
    cata_name=cata_info[1]
    print(cata_name)
    
cata_name2=str(input('Input the name of catalog :'))





result3 = Vizier.query_region(coord.SkyCoord
                              (ra=ra0_deg, dec=dec0_deg,unit=(u.deg, u.deg),
                              frame='icrs'),radius=r2_deg*u.deg,
                              catalog=[cata_name2])
n_row3=len(result3[0])
#    key_result3=result3.keys()[0]
#    krs=key_result3.split('/',-1)
cata_name3=cata_name2.replace('/','')
print('replace catalog name as: ',cata_name3)
input("Press Enter to continue...")


file_result3=obj_name+'_cata_'+cata_name3+'_list_area_r'+str(int(r2_arcm))+'arcm.txt'
ascii.write(result3[0],file_result3,delimiter='|',overwrite=True) 


data_table=result3[0]
header=data_table.colnames
print(header)
    #data_table['']

df_cat3=pd.DataFrame(np.array(data_table))
ra_cat3=df_cat3['RAJ2000']
dec_cat3=df_cat3['DEJ2000']
dx=abs(ra_cat3-ra0_deg)
dy=abs(dec_cat3-dec0_deg)
dxy=np.sqrt(dx**2+dy**2)
df_cat3['dxy_deg']=dxy
    
ra_hhmmss=['']*n_row3
dec_ddmmss=['']*n_row3

# coordinate convertor
# https://docs.astropy.org/en/stable/coordinates/
for i in range(n_row3):
    radec_deg=SkyCoord(ra_cat3[i],dec_cat3[i],frame='icrs',unit='deg')
    radec_sexe=radec_deg.to_string('hmsdms').split(' ',-1)
    print(radec_sexe[0],radec_sexe[1])
    ra_hhmmss[i]=radec_sexe[0]
    dec_ddmmss[i]=radec_sexe[1]

df_cat3['RA_hhmmss']=ra_hhmmss
df_cat3['DEC_ddmmss']=dec_ddmmss


print()
    #df_cat3new=df_cat3[['RAJ2000','DEJ2000','dxy_deg','RA_hhmmss','DEC_ddmmss','Vmag','rmag']].sort_values(by=['dxy_deg']).reset_index(drop=True)
df_cat3new=df_cat3[['RAJ2000','DEJ2000','dxy_deg','RA_hhmmss','DEC_ddmmss','Rmag']].sort_values(by=['dxy_deg']).reset_index(drop=True)
print(df_cat3new)
print()

file_cata=obj_name+'_list_area_r'+str(int(r2_arcm))+'arcm.txt'
    #ascii.write(df_cat3new,file_cata,delimiter='|')   
df_cat3new.to_csv(file_cata,index=True)

print('-------')
r4_as=15
print('... will circle each source with radius =',r4_as,'[as] ...')
#input("Press Enter to continue...")

file_reg_fk5=obj_name+'_'+cata_name3+'_list_area_r'+str(int(r2_arcm))+'arcm_r'+str(int(r4_as))+'_as_fk5.reg'


f_reg=open(file_reg_fk5,'w')

f_reg.write('# Region file format: DS9 version 4.1\n')
f_reg.write('global color=yellow dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f_reg.write('fk5\n')

ra_cat3new=df_cat3new['RA_hhmmss']
dec_cat3new=df_cat3new['DEC_ddmmss']



for i in range(n_row3):
    if i==0:
        txt_coord='circle('+str(ra_cat3new[i])+','+str(dec_cat3new[i])+','+str(r4_as)+'") # color=white'
    else:
        txt_coord='circle('+str(ra_cat3new[i])+','+str(dec_cat3new[i])+','+str(r4_as)+'")'
    f_reg.write(txt_coord+'\n')
 
f_reg.close()

