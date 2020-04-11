#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:15:21 2019

@author: altsai
"""


#import os
#import sys
#import shutil
import numpy as np
#import csv
#import time
#import math
import pandas as pd
import matplotlib.pyplot as plt

nn=30
'''
df_201709_me=pd.read_csv('Rmag_InstMag/Rmag_InstMag/annu_w0_2017/Rmag_w0_2017_txt/Rmag_3C66A_w0_201709.txt',delimiter=',').iloc[0:nn]
print(df_201709_me)
JD1=df_201709_me['JD'].map('{:.5f}'.format).astype(np.float64)
print(JD1)
R1=df_201709_me['Rmag']
print(R1)
eR1=df_201709_me['ErrorRmag']
n1=len(R1)
print(n1)

#df_20170912_manash=pd.read_csv('old_results/final_results_Sep_Dec_2017/g0219r_LuS_180110.dat',usecols=[1,3],sep='\s+',header=None,names=['JD','Rmag','ErrorRmag'])
df_20170912_manash=pd.read_csv('old_results/final_results_Sep_Dec_2017/g0219r_LuS_180110.dat',sep='\s+',header=None)
#print(df_20170912_manash)
df_201709_manash=df_20170912_manash.iloc[0:nn]
print(df_201709_manash)
JD0=df_201709_manash[0]
R0=df_201709_manash[1]
print(R0)
eR0=df_201709_manash[2]

dJD=JD1-JD0
dR=R1-R0
deR=eR1-eR0

print(dR)
print(deR)

print(np.mean(dR))
'''

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
list_data_manash_2017=['g0219r_LuS_180110.dat','g0235r_LuS_180110.dat','g0716r_LuS_180110.dat','g0827r_LuS_180110.dat','g0829r_LuS_180110.dat','g0836r_LuS_180110.dat','g0954r_LuS_180110.dat','g1101r_LuS_180110.dat','g1156r_LuS_180110.dat','g1219r_LuS_180110.dat','g1226r_LuS_180110.dat','g1253r_LuS_180110.dat','g1652r_LuS_180110.dat','g1739r_LuS_180110.dat','g1807r_LuS_180110.dat','g2155r_LuS_180110.dat','g2200r_LuS_180110.dat','g2230r_LuS_180110.dat','g2251r_LuS_180110.dat','g2344r_LuS_180110.dat']
#list_data_manash_match=list(list_data_manash_2017[i] for i in [0,1,3,4,5,12,13,14,16,17,18,19])
list_data_manash_match=list(list_data_manash_2017[i] for i in [0,1,2,3,4,5,12,13,14,15,16,17,18,19])
list_source=['3C66A', 'AO0235+16', 'S5_0716+71', 'PKS0735+17', 'OJ248', 'OJ49', '4C71-07', 'OJ287', 'S4_0954+65', 'Mkn421', '4C29-45', 'ON231', '3C273', '3C279', 'KS1510-08', 'DA406', '4C38-41', '3C345', 'Mkn501', '4C51-37', '3C371', 'PKS2155-304', 'L-Lacertae', 'CTA102', '3C454-3', 'ES2344+514']
list_source_match=list(list_source[i] for i in [0,1,2,4,5,6,18,19,20,21,22,23,24,25])
#list_source_match=list(list_source[i] for i in [0,1,4,5,6,18,19,20,22,23,24,25])
#list_source_match=['3C66A', 'AO0235+16', 'S5_0716+71', 'OJ248', 'OJ49', '4C71-07', 'S4_0954+65', 'Mkn421', '4C29-45', 'ON231', '3C273', '3C279', 'Mkn501', '4C51-37', '3C371', 'PKS2155-304', 'L-Lacertae', 'CTA102', '3C454-3', 'ES2344+514']
#list_source_match=['3C66A', 'AO0235+16', 'S5_0716+71', 'OJ248', 'OJ49', '4C71-07', 'S4_0954+65', 'Mkn421', '4C29-45', 'ON231', '3C273', '3C279', 'Mkn501', '4C51-37', '3C371', 'PKS2155-304', 'L-Lacertae', 'CTA102', '3C454-3', 'ES2344+514']

#R1=np.array([0]*nn)
#R0=np.array([0]*nn)
n_data=len(list_data_manash_match)

fig,axs=plt.subplots(3,5,figsize=(16,12))
fig.subplots_adjust(hspace=.5,wspace=0.4)
axs=axs.ravel()

for i in range(n_data):
    ii=n_data+i
    file_manash=list_data_manash_match[i]
    df_manash=pd.read_csv('old_results/final_results_Sep_Dec_2017/'+file_manash,sep='\s+',header=None).iloc[0:nn]
    JD0=df_manash[0]
    R0=df_manash[1]
#    print(R0)
    eR0=df_manash[2]
    
    source_name=list_source_match[i]
    print()
    print(source_name,'|',file_manash)
    file_match='Rmag_'+source_name+'.txt'
    print('w0')
    df_w0_match=pd.read_csv('Rmag_InstMag/annu_w0_2017/Rmag_w0_2017_txt/'+file_match,delimiter=',').iloc[0:nn]
    JD1=df_w0_match['JD'].map('{:.5f}'.format).astype(np.float64)
    print(JD1)
    R1=df_w0_match['Rmag']
    print(R1)
    eR1=df_w0_match['ErrorRmag']
    print(eR1)
    
#    dJD=JD1-JD0
#    dR=R1-R0
#    deR=eR1-eR0
#    mean_dR=np.mean(dR)
#    mean_deR=np.mean(deR)

#    print('    dJD     d(R)  d(errorR)')
#    for j in range(nn):
#        print(j,'%.5f' %dJD[j],'%.4f' %dR[j],'%.4f' %deR[j])

#    print(source_name,'|',file_manash)
#    print('mean_dR=','%.4f' %mean_dR,'mean_deR=','%.4f' %mean_deR)

#    plt.figure()
    
    print('w1')
    df_w1_match=pd.read_csv('Rmag_InstMag/annu_w1_2017/Rmag_w1_2017_txt/'+file_match,delimiter=',').iloc[0:nn]
    JD2=df_w1_match['JD'].map('{:.5f}'.format).astype(np.float64)
    print(JD2)
    R2=df_w1_match['Rmag']
    print(R2)
    eR2=df_w1_match['ErrorRmag']
    print(eR2)
    
    
    axs[i].errorbar(JD0,R0,yerr=eR0,linestyle='--',label='manash',lw=1)
    axs[i].errorbar(JD1,R1,yerr=eR1,linestyle='--',label='me(no weighting)',lw=1)  
    axs[i].errorbar(JD2,R2,yerr=eR2,linestyle='--',label='me(weighting)',lw=1)
    axs[i].set_xlabel('JD')
    axs[i].set_ylabel('Rmag')
    axs[i].set_title(source_name)
    axs[i].invert_yaxis()
    axs[i].legend(loc='best')
    
    '''
    axs[ii].errorbar(JD0,R0,yerr=eR0,label='manash',lw=1)
    axs[ii].errorbar(JD1,R1,yerr=eR1,label='me(no weighting)',lw=1)  
#    axs[i].errorbar(JD2,R2,yerr=eR2,label='me(weighting)',lw=1)
    axs[ii].set_xlabel('JD')
    axs[ii].set_ylabel('Rmag')
    axs[ii].set_title(source_name)
    axs[ii].invert_yaxis()
    axs[ii].legend(loc='best')
    '''
#    plt.errorbar(JD1,R1,yerr=eR1,label='me')  
#    plt.errorbar(JD0,R0,yerr=eR0,label='manash')  
#    plt.title('tttt')
#    plt.xlabel('JD')
#    plt.ylabel('Rmag)
#    plt.legend(loc='best')
#    plt.gca().invert_yaxis()
#    plt.show()

plt.savefig('diff_me_manash_201709.pdf')   
    
# 3C66A | g0219r_LuS_180110.dat | mean_dR= -0.0626 mean_deR= 0.0138
# AO0235+16 | g0235r_LuS_180110.dat | mean_dR= -0.3176 mean_deR= -0.4278 (X)
# OJ248 | g0827r_LuS_180110.dat | mean_dR= 0.1572 mean_deR= -0.0100 (?)
# OJ49 | g0829r_LuS_180110.dat | mean_dR= 0.0618 mean_deR= -0.0193
# 4C71-07 | g0836r_LuS_180110.dat | mean_dR= -0.0473 mean_deR= 0.0170
# Mkn501 | g1652r_LuS_180110.dat | mean_dR= -0.5546 mean_deR= 0.0018 (X)
# 4C51-37 | g1739r_LuS_180110.dat | mean_dR= -0.1860 mean_deR= 0.0091 (?)
# 3C371 | g1807r_LuS_180110.dat | mean_dR= -0.2232 mean_deR= -0.0327 (X)
# L-Lacertae | g2200r_LuS_180110.dat | mean_dR= -0.0072 mean_deR= -0.0199
# CTA102 | g2230r_LuS_180110.dat | mean_dR= -0.1404 mean_deR= -0.0101
# 3C454-3 | g2251r_LuS_180110.dat | mean_dR= 0.1606 mean_deR= 0.0006
# ES2344+514 | g2344r_LuS_180110.dat | mean_dR= -0.2063 mean_deR= -0.0163 (X)
    
