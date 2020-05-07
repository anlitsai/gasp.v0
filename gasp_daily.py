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
#print(list_targetname)

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
list_out3=[]

file_out1='gasp_daily1.txt'
if os.path.exists(file_out1):
    os.remove(file_out1)

file_out2='gasp_daily2.txt'
if os.path.exists(file_out2):
    os.remove(file_out2)
    
file_out3='gasp_daily3.txt'
if os.path.exists(file_out3):
    os.remove(file_out3)
    
    

with open(file_out1,'w') as f_out1, open(file_out2,'w') as f_out2, open(file_out3,'w') as f_out3:
   
    for i in list_dailylist:
        #print(i)
        date=[i.split('/',2)[1]][0][3:11]
        #print(date)
        fitsname=[i.split('/',-1)[-1]][0]
        #print(fitsname)
        if "-S001" in fitsname:
            targetname=fitsname.split('-S001',2)[0]
        elif "@" in fitsname:
            targetname=fitsname.split('-20',2)[0]    
            #print(targetname)

        ###### date:fitsname ######
        date_fitsname=date+':'+fitsname
        #print(date_fitsname)
#        f_out1.write(date_fitsname+'\n')
        list_out1.append(date_fitsname)
        
        ###### date:targetname ###### 
        date_targetname=date+':'+targetname
        #print(date_targetname)
        #f_out2.write(date_fitsname+'\n')
#        f_out2.write(date_targetname+'\n')
        list_out2.append(date_targetname)

        ###### remove duplicate ######
        if date_targetname not in list_out3:
#            print(date_targetname)
            list_out3.append(date_targetname)
            f_out3.write(date_targetname+'\n')

        ###### replace targetname ######
        
