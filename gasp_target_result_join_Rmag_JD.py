#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:23:12 2020

@author: altsai
"""

import os
import sys
#import shutil
#import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#import matplotlib.axes as ax

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0


target_file='check_science_target_list.txt'
cmd_search_target='cat '+target_file
#print(cmd_search_target)
target_list=os.popen(cmd_search_target,"r").read().splitlines() #[0].split(' ')
#print(target_list)
n_target=len(target_list)
#print(n_target)

result_file='gasp_target_result_join.txt'

df=pd.read_csv('gasp_target_result_join.txt',delimiter='\t')
#print(df)


fntsz=12

pdf_pages=PdfPages('gasp_target_result_join_Rmag_JD.pdf')

for i in target_list:
#    print(i)
    df_target=df[df['Target']==i]
    JD=df_target['JulianDay']
    mag=df_target['Mag']
    mag_err=df_target['Mag_err']
    filter_list=df_target['Filter'].tolist()
    filter_band=filter_list[0].capitalize()
    
    fig=plt.figure(figsize=(8,6),dpi=200)
    plt.errorbar(JD,mag,yerr=mag_err,fmt='.',color='red')
    ttl=i
    plt.title(ttl,fontsize=fntsz)
    plt.xlabel('JD',fontsize=fntsz)
    yname=filter_band+'mag'
    plt.ylabel(yname,fontsize=fntsz)
    plt.xticks(fontsize=fntsz)
    plt.yticks(fontsize=fntsz)
    plt.gca().invert_yaxis()
#    plt.legend(loc='best')
    pdf_pages.savefig(fig)

pdf_pages.close()
    
    



