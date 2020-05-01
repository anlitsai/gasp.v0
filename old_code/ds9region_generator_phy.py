#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 01:22:55 2019

@author: altsai
"""



import os
import sys
#import shutil
#import re
import numpy as np
#import numpy
#from astropy.io import fits
#import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#import glob
#import subprocess
#from scipy import interpolate
#from scipy import stats
#from scipy.interpolate import griddata
#from time import gmtime, strftime
#import pandas as pd
#from datetime import datetime
from astropy.io import ascii
#from operator import itemgetter

#file_sourcelist_coord_px=input("Enter file name (sourceListByAP.dat): ")
file_sourcelist_coord_px='sourceListByAPT.dat'
file_sourcelist_name=file_sourcelist_coord_px.split('.',2)[0]
print(file_sourcelist_name)


file_reg_phy=file_sourcelist_name+'_phy.reg'
print('will write to : '+file_reg_phy)
if os.path.exists(file_reg_phy):
    os.remove(file_reg_phy)
f_reg=open(file_reg_phy,'w')

f_reg.write('# Region file format: DS9 version 4.1\n')
f_reg.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f_reg.write('physical\n')
#sys.exit(0)
array_sourcelist_coord_px=ascii.read(file_sourcelist_coord_px)
print(array_sourcelist_coord_px)
n_line=len(array_sourcelist_coord_px)
#print(n_line)


xydxy=np.zeros((3,n_line))

for i in range(n_line):
    x=array_sourcelist_coord_px[i][0]
    y=array_sourcelist_coord_px[i][1]
    xydxy[0][i]=x
    xydxy[1][i]=y
    dx=abs(x-1024.)
    dy=abs(y-1024.)
    dxy=np.sqrt(dx**2+dy**2)
    xydxy[2][i]=dxy

print(xydxy)
#sys.exit(0)
print('--------------')

sorted_xydxy=xydxy[:,xydxy[-1].argsort()]

#xydxy=sorted(xydxy,key=itemgetter(2))
print(sorted_xydxy)
#sys.exit(0)



for i in range(n_line):
    if i==0:
        txt_coord='circle('+str(sorted_xydxy[0][i])+','+str(sorted_xydxy[1][i])+','+'15) # color=red'
    else:
        txt_coord='circle('+str(sorted_xydxy[0][i])+','+str(sorted_xydxy[1][i])+','+'15)'
    f_reg.write(txt_coord+'\n')
    
f_reg.close()


