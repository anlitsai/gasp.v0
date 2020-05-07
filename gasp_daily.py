#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 00:16:10 2020

@author: altsai
"""

import os
import sys
import shutil
import re
import numpy as np
import pandas as pd


file_targetname='check_science_target_list.txt'
cmd_targetname="cat "+file_targetname
list_targetname=os.popen(cmd_targetname,"r").read().splitlines()
print(list_targetname)

#sys.exit(0)

file_fits='gasp_daily.list1'
#file_datetarget='gasp_daily_target.txt'
#if os.path.exists(file_datetarget):
#    os.remove(file_datetarget)

cmd_dailyfile="cat "+file_fits
list_dailylist=os.popen(cmd_dailyfile,"r").read().splitlines()

list_datetarget=[]

# ====================================
# output >>>> date:targetname
# ====================================
list_out1=[]
list_out2=[]

file_out1='gasp_daily1.txt'
if os.path.exists(file_out1):
    os.remove(file_out1)
    
f_out1=open(file_out1,'w')

for i in list_dailylist:
#    print(i)
    date=[i.split('/',2)[1]][0][3:11]
#    print(date)
    fitsname=[i.split('/',-1)[-1]][0]
#    print(fitsname)
    if "-S001" in fitsname:
        targetname=fitsname.split('-S001',2)[0]
    elif "@" in fitsname:
        targetname=fitsname.split('-20',2)[0]    
#    print(targetname)
    date_target=date+':'+targetname
#    print(date_target)
    f_out1.write(date_target+'\n')
'''
    ###### remove duplicate ######
    if date_target not in list_out1:
        print(date_target)
        list_out1.append(date_target)
        f_out1.write(date_target+'\n')
'''


f_out1.close

#print(list_out1)

#list_out1



'''
list_datetarget2=[]

f_listout=open(file_datetarget,'w')

for i in list_datetarget:
    if i not in list_datetarget2:
        list_datetarget2.append(i) 
        f_listout.write(i+'\n')

f_listout.close()


print(list_datetarget2)'''