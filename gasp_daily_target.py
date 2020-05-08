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


file_targetname='all_science_target_list.txt'
cmd_targetname="cat "+file_targetname
list_targetname=os.popen(cmd_targetname,"r").read().splitlines()
#print(list_targetname)

file_targetrename='all_science_target_list_rename.txt'
cmd_targetrename="cat "+file_targetrename
list_targetrename=os.popen(cmd_targetrename,"r").read().splitlines()
#print(list_targetrename)


#sys.exit(0)

file_fits='gasp_daily.list1'
#file_datetarget='gasp_daily_target.txt'
#if os.path.exists(file_datetarget):
#    os.remove(file_datetarget)

cmd_dailyfile="cat "+file_fits
list_dailylist=os.popen(cmd_dailyfile,"r").read().splitlines()

list_datetarget=[]

list_filter=['B_','V_','R_','I_']



# ====================================
# output >>>> date:targetname
# ====================================
list_out1=[]
list_out2=[]
list_out3=[]
list_out4=[]

file_out1='gasp_daily1.txt'
if os.path.exists(file_out1):
    os.remove(file_out1)

file_out2='gasp_daily2.txt'
if os.path.exists(file_out2):
    os.remove(file_out2)
    
file_out3='gasp_daily3.txt'
if os.path.exists(file_out3):
    os.remove(file_out3)

file_out4='gasp_daily_target.txt'
if os.path.exists(file_out4):
    os.remove(file_out4)    
    
with open(file_out1,'w') as f_out1, open(file_out2,'w') as f_out2, open(file_out3,'w') as f_out3, open(file_out4,'w') as f_out4:

    f_out4.write('YYYYMMDD\tFilter\tTarget\n')
    
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
        
        for filterband in list_filter:
            if filterband in fitsname:
                filtername=filterband.split('_',2)[0]
                #print(filtername)

        ###### date:fitsname ######
        date_fitsname=date+':'+fitsname
        #print(date_fitsname)
        f_out1.write(date_fitsname+'\n')
        list_out1.append(date_fitsname)
        
        ###### date:targetname ###### 
        #date_targetname=date+':'+targetname+':'+filtername
        date_targetname=date+':'+filtername+':'+targetname
        #print(date_targetname)
        #f_out2.write(date_fitsname+'\n')
        f_out2.write(date_targetname+'\n')
        list_out2.append(date_targetname)

        ###### remove duplicate ######
        if date_targetname not in list_out3:
#            print(date_targetname)
            list_out3.append(date_targetname)
            f_out3.write(date_targetname+'\n')

        ###### replace targetname & remove duplicate ######
        if targetname in list_targetname:
            target_idx=[j for j,elem in enumerate(list_targetname) if targetname in elem]
#            print(target_idx)
            targetrename=list_targetrename[target_idx[0]]
#            print(targetrename)
            date_targetrename=date+'\t'+filtername+'\t'+targetrename
            if date_targetrename not in list_out4:
                list_out4.append(date_targetrename)
                f_out4.write(date_targetrename+'\n')

