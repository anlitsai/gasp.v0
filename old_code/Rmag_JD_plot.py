#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:43:22 2019

@author: altsai
"""

import pandas as pd
import matplotlib.pyplot as plt

'''
altsai@hyperion2 [3C345]$ find ./test* |grep txt
./test2/circle_r=fwhmx3/Rmag_aperture_3C345_fwhm3.txt
./test2/circle_r=fwhmx5/Rmag_aperture_3C345_fwhm5.txt
./test2/circle_r=fwhmx4/Rmag_aperture_3C345_fwhm4.txt
./test2/circle_r=fwhmx2/Rmag_aperture_3C345_fwhm2.txt
'''

file1='./Rmag_InstMag/3C345/test2/circle_r=fwhmx2/Rmag_aperture_3C345_fwhm2.txt'
file2='./Rmag_InstMag/3C345/test2/circle_r=fwhmx3/Rmag_aperture_3C345_fwhm3.txt'
file3='./Rmag_InstMag/3C345/test2/circle_r=fwhmx4/Rmag_aperture_3C345_fwhm4.txt'
file4='./Rmag_InstMag/3C345/test2/circle_r=fwhmx5/Rmag_aperture_3C345_fwhm5.txt'

df1=pd.read_csv(file1,sep="|")
df2=pd.read_csv(file2,sep="|")
df3=pd.read_csv(file3,sep="|")
df4=pd.read_csv(file4,sep="|")
JD=df1['JD']
Rmag1=df1['Rmag']
Rmag2=df2['Rmag']
Rmag3=df3['Rmag']
Rmag4=df4['Rmag']


plt.figure(figsize=(8,6),dpi=100)
plt.plot(JD,Rmag1,label='fwhm x2')
plt.plot(JD,Rmag2,label='fwhm x3')
plt.plot(JD,Rmag3,label='fwhm x4')
plt.plot(JD,Rmag4,label='fwhm x5')
plt.xlabel('JD')
plt.ylabel('Rmag')
plt.gca().invert_yaxis()
plt.title('3C345')
plt.legend(loc='best')
plt.show()
#plt.savefig('./Rmag_InstMag/3C345/test2/Rmag_JD/3C345_fwhm2345.png')

