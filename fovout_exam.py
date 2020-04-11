#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:27:01 2019

@author: altsai
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from datetime import *
import datetime

with open('check_science_target_list.txt') as f:
    target_list=f.read().splitlines()
f.close()
print(target_list)
n_target=len(target_list)
idx_target=np.array(range(n_target))
print(idx_target)

n_target_fovout_i=np.array([0]*n_target)
n_target_source_i=np.array([0]*n_target)
percent_target_fovout_i=np.array([0.0]*n_target)




df_all=pd.read_csv('gasp_target_fitsheader_info_exclude_baddata_join.txt',sep='|')
#print(df_all)


df_bad=pd.read_csv('bad_data_note.txt',sep='\t')
#print(df_bad)


#print(df_bad.Note)
df_fovout=df_bad[df_bad.Note.str.contains('fovout')]
#print(df_fovout)

#print(df_fovout.filename)

date1='2018-04-01'
date2='2019-12-31'
month_list1 = [i.strftime("%Y-%m") for i in pd.date_range(start=date1, end=date2, freq='MS')]
#print(month_list1)
n_month=len(month_list1)

month_list2= [i.strftime("%Y%m") for i in pd.date_range(start=date1, end=date2, freq='MS')]
#print(month_list2)

n_month_fovout_i=np.array([0]*n_month)
n_month_source_i=np.array([0]*n_month)
percent_month_fovout_i=np.array([0.0]*n_month)

print('Month','Number','%')
for i in range(n_month):
    month1=month_list1[i]
    month2=month_list2[i]
    df_month_source_i=df_all[df_all.DateObs.str.contains(month1)]
    n_month_source_i[i]=len(df_month_source_i)
    df_month_fovout_i=df_fovout[df_fovout.filename.str.contains(month2)]
#    print(df_target_fovout_i)
    n_month_fovout_i[i]=len(df_month_fovout_i)
    percent_month_fovout_i[i]=n_month_fovout_i[i]/n_month_source_i[i]*100
    print(month1,n_month_fovout_i[i],n_month_source_i[i],'%.2f' %percent_month_fovout_i[i])

plt.figure(figsize=(18,22))
plt.subplot(4,2,1)
plt.bar(month_list1,n_month_fovout_i)
plt.xticks(rotation='vertical')
plt.ylabel('number of FOV-out per month')
#plt.savefig('fovout_month_number.pdf')
#plt.show()


plt.subplot(4,2,2)
plt.bar(month_list1,percent_month_fovout_i)
plt.xticks(rotation='vertical')
plt.ylabel('% of FOV-out per month')
#plt.savefig('fovout_month_percent.pdf')
#plt.show()




print('Target','Number','%')
for i in range(n_target):
    target=target_list[i]
    targ=target[:7]
    df_target_source_i=df_all[df_all.Filename.str.contains(targ)]
    n_target_source_i[i]=len(df_target_source_i)
    df_target_fovout_i=df_fovout[df_fovout.filename.str.contains(targ)]
#    print(df_target_fovout_i)
    n_target_fovout_i[i]=len(df_target_fovout_i)
    percent_target_fovout_i[i]=n_target_fovout_i[i]/n_target_source_i[i]*100
    print(target,n_target_fovout_i[i],n_target_source_i[i],'%.2f' %percent_target_fovout_i[i])

n_target_fovout=sum(n_target_fovout_i)    
print(n_target_fovout)

n_all=sum(n_target_source_i)
print(n_all)
    


#print(nn)


plt.subplot(4,2,3)
plt.bar(idx_target,n_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
#plt.xticks([])
plt.ylabel('number of FOV-out per target')
#plt.savefig('fovout_target_number.pdf')
#plt.show()

plt.subplot(4,2,4)
plt.bar(idx_target,percent_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
#plt.xticks([])
plt.ylabel('% of FOV-out per target')
#plt.savefig('fovout_target_percent.pdf')
#plt.show()



print('Target @ 2019/11','Number','%')
for i in range(n_target):
    target=target_list[i]
    targ=target[:7]
    df_target_source_i=df_all[(df_all.Filename.str.contains(targ)) & (df_all.DateObs.str.contains('2019-11'))]
    n_target_source_i[i]=len(df_target_source_i)
    df_target_fovout_i=df_fovout[(df_fovout.filename.str.contains(targ)) & (df_fovout.filename.str.contains('201911'))]
#    print(df_target_fovout_i)
    n_target_fovout_i[i]=len(df_target_fovout_i)
    percent_target_fovout_i[i]=n_target_fovout_i[i]/n_target_source_i[i]*100
    print(target,n_target_fovout_i[i],n_target_source_i[i],'%.2f' %percent_target_fovout_i[i])

n_target_fovout=sum(n_target_fovout_i)    
print(n_target_fovout)

n_all=sum(n_target_source_i)
print(n_all)
    


#print(nn)

plt.subplot(4,2,5)
plt.bar(idx_target,n_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
#plt.xticks([])
plt.ylabel('number of FOV-out per target @ 2019/11')
#plt.savefig('fovout_target_number.pdf')
#plt.show()

plt.subplot(4,2,6)
plt.bar(idx_target,percent_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
#plt.xticks([])
plt.ylabel('% of FOV-out per target @ 2019/11')
#plt.savefig('fovout_target_percent.pdf')
#plt.show()



print('Target @ 2019/12','Number','%')
for i in range(n_target):
    target=target_list[i]
    targ=target[:7]
    df_target_source_i=df_all[(df_all.Filename.str.contains(targ)) & (df_all.DateObs.str.contains('2019-12'))]
    n_target_source_i[i]=len(df_target_source_i)
    df_target_fovout_i=df_fovout[(df_fovout.filename.str.contains(targ)) & (df_fovout.filename.str.contains('201912'))]
#    print(df_target_fovout_i)
    n_target_fovout_i[i]=len(df_target_fovout_i)
    percent_target_fovout_i[i]=n_target_fovout_i[i]/n_target_source_i[i]*100
    print(target,n_target_fovout_i[i],n_target_source_i[i],'%.2f' %percent_target_fovout_i[i])

n_target_fovout=sum(n_target_fovout_i)    
print(n_target_fovout)

n_all=sum(n_target_source_i)
print(n_all)
    


#print(nn)

plt.subplot(4,2,7)
plt.bar(idx_target,n_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
plt.ylabel('number of FOV-out per target @ 2019/12')
#plt.savefig('fovout_target_number.pdf')
#plt.show()

plt.subplot(4,2,8)
plt.bar(idx_target,percent_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
plt.ylabel('% of FOV-out per target @ 2019/12')
#plt.savefig('fovout_target_percent.pdf')
#plt.show()




plt.savefig('fovout_exam.pdf')
plt.show()



print('---------------')
print('December       ')
print('---------------')

date1='20191201'
date2='20191209'
date3='20191231'

date_1 = datetime.date(int(date1[0:4]),int(date1[4:6]),int(date1[6:8]))
date_2 = datetime.date(int(date2[0:4]),int(date2[4:6]),int(date2[6:8]))
date_3 = datetime.date(int(date3[0:4]),int(date3[4:6]),int(date3[6:8]))

print('---------------')
print('December 1-9   ')
print('---------------')

print('Target @ 2019/12/1-9','Number','%')
for i in range(n_target):
    target=target_list[i]
    targ=target[:7]
    df_target_source_i=df_all[(df_all.Filename.str.contains(targ)) & (df_all.DateObs.str.contains('2019-12-0'))]
#    df_target_source_i=df_all[(df_all.Filename.str.contains(targ)) & (df_all['DateObs']>=date_1) & (df_all['DateObs']<date_2)]
#    df_target_source_i=df_all[(df_all['Filename'].str.contains(targ)) & (df_all['DateObs']>=date_1) & (df_all['DateObs']<date_2)]
    n_target_source_i[i]=len(df_target_source_i)
    df_target_fovout_i=df_fovout[(df_fovout.filename.str.contains(targ)) & (df_fovout.filename.str.contains('2019120'))]
#    print(df_target_fovout_i)
    n_target_fovout_i[i]=len(df_target_fovout_i)
    percent_target_fovout_i[i]=n_target_fovout_i[i]/n_target_source_i[i]*100
    print(target,n_target_fovout_i[i],n_target_source_i[i],'%.2f' %percent_target_fovout_i[i])

n_target_fovout=sum(n_target_fovout_i)    
print(n_target_fovout)

n_all=sum(n_target_source_i)
print(n_all)
    


#print(nn)
plt.figure(figsize=(16,12))

plt.subplot(2,2,1)
plt.bar(idx_target,n_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
plt.ylabel('number of FOV-out per target @ 2019/12/1-9')
#plt.savefig('fovout_target_number.pdf')
#plt.show()

plt.subplot(2,2,2)
plt.bar(idx_target,percent_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
plt.ylabel('% of FOV-out per target @ 2019/12/1-9')
#plt.savefig('fovout_target_percent.pdf')
#plt.show()

#sys.exit(0)

print('---------------')
print('December 10-   ')
print('---------------')

print('Target @ 2019/12/10-24','Number','%')
for i in range(n_target):
    target=target_list[i]
    targ=target[:7]
    df_target_source_i=df_all[(df_all.Filename.str.contains(targ)) & (df_all.DateObs.str.contains('2019-12-1'))]
#    df_target_source_i2=df_target_source_i1[(df_all.DateObs>=date_2) & (df_all.DateObs<=date_3)]
    n_target_source_i[i]=len(df_target_source_i)
    df_target_fovout_i=df_fovout[(df_fovout.filename.str.contains(targ)) & (df_fovout.filename.str.contains('2019121'))]
#    print(df_target_fovout_i)
    n_target_fovout_i[i]=len(df_target_fovout_i)
    percent_target_fovout_i[i]=n_target_fovout_i[i]/n_target_source_i[i]*100
    print(target,n_target_fovout_i[i],n_target_source_i[i],'%.2f' %percent_target_fovout_i[i])

n_target_fovout=sum(n_target_fovout_i)    
print(n_target_fovout)

n_all=sum(n_target_source_i)
print(n_all)
    


#print(nn)

plt.subplot(2,2,3)
plt.bar(idx_target,n_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
plt.ylabel('number of FOV-out per target @ 2019/12/10-')
#plt.savefig('fovout_target_number.pdf')
#plt.show()

plt.subplot(2,2,4)
plt.bar(idx_target,percent_target_fovout_i)
plt.xticks(idx_target,target_list, rotation='vertical')
plt.ylabel('% of FOV-out per target @ 2019/12/10-')
#plt.savefig('fovout_target_percent.pdf')
#plt.show()

plt.savefig('fovout_201912.pdf')
plt.show()
