#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:04:44 2019

@author: altsai
"""


import os
import sys
import shutil
import numpy as np
#import csv
#import time
#import math
import pandas as pd
import matplotlib.pyplot as plt




'''
 0|0219+428|3C66A      |g0219r_LuS_180110.dat|
 1|0235+164|AO0235+16  |g0235r_LuS_180110.dat|
 2|0716+714|S5_0716+71 |g0716r_LuS_180110.dat|xo
 3|0735+178|PKS0735+17
 4|0827+243|OJ248      |g0827r_LuS_180110.dat|
 5|0829+046|OJ49       |g0829r_LuS_180110.dat|
 6|0836+710|4C71-07    |g0836r_LuS_180110.dat|
 7|0851+202|OJ287
 8|0954+658|S4_0954+65 |g0954r_LuS_180110.dat|
 9|1101+384|Mkn421     |g1101r_LuS_180110.dat|x
10|1156+295|4C29-45    |g1156r_LuS_180110.dat|x
11|1219+285|ON231      |g1219r_LuS_180110.dat|x
12|1226+023|3C273      |g1226r_LuS_180110.dat|x
13|1253-055|3C279      |g1253r_LuS_180110.dat|x
14|1510-089|KS1510-08
15|1611+343|DA406
16|1633+382|4C38-41
17|1641+399|3C345
18|1652+398|Mkn501     |g1652r_LuS_180110.dat|
19|1739+522|4C51-37    |g1739r_LuS_180110.dat|
20|1807+698|3C371      |g1807r_LuS_180110.dat|
21|2155-304|PKS2155-304|g2155r_LuS_180110.dat|+o
22|2200+420|L-Lacertae |g2200r_LuS_180110.dat|
23|2230+114|CTA102     |g2230r_LuS_180110.dat|
24|2251+158|3C454-3    |g2251r_LuS_180110.dat|
25|2344+514|ES2344+514 |g2344r_LuS_180110.dat|
'''

file_list_in='gasp_refStar_radec_annu.txt'
df_list_in=pd.read_csv(file_list_in,sep='|',usecols=[1,2])
obj_name=df_list_in['ObjectName']
iau_name=df_list_in['IAU_Name']
list_obj=pd.unique(obj_name)
print(list_obj)
list_iau=pd.unique(iau_name)
print(list_iau)

n_obj=len(list_obj)



path='./Rmag_InstMag/annu_w1_2017/Rmag_w1_2017_txt/'
#path=sys.argv[1]

cmd_search_txt='find '+path+' | grep txt| cut -d / -f5'
print(cmd_search_txt)
list_txt=os.popen(cmd_search_txt,"r").read().splitlines() #[0].split(',')
print(list_txt)

for i in range(n_obj):
    print(i,list_obj[i],list_iau[i])
    src=path+'Rmag_'+list_obj[i]+'.txt'
    dst='Rmag_InstMag/gasp_dat/gasp_'+list_iau[i]+'.dat'
#    os.rename(src,dst)
    shutil.copyfile(src,dst)
