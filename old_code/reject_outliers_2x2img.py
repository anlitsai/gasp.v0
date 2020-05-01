#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 02:15:57 2019

@author: altsai
"""


import numpy as np
from scipy.interpolate import griddata

'''
def reject_outliers_2x2img(data3d,threshold=3.):
    median_value=np.median(np.mean(data3d,axis=(1,2)))
    #print(median_value)
    diff=abs(data3d-median_value)
    #print(diff)
    med_diff=np.median(diff)
    #print(med_diff)
    print('...remove outlier...')
    if med_diff == 0:
        s=0
    else:
        s=diff/med_diff
        #print(s)
    #np.delete(data3d,,axis=0)
    data3d_keep=np.where(s<threshold,data3d,np.nan)
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data3d_keep
'''

def reject_outliers_2x2img(data3d,threshold=3.):
    median_value=np.median(np.mean(data3d,axis=(1,2)))
    #print(median_value)
    diff=abs(data3d-median_value)
    #print(diff)
    med_diff=np.median(diff)
    #print(med_diff)
    print('...remove outlier...')
    if med_diff == 0:
        s=0
    else:
        s=diff/med_diff
        #print(s)
    #np.delete(data3d,,axis=0)
    data3d_keep=np.where(s<threshold,data3d,np.nan)
#    data3d_interp=data3d_keep.interp
    print('...interpolation...')
    #indexes=np.arange(data3d_keep.shape[0])
    #good=np.isfinite(data3d_keep).al(axis=(1,2))
    #f=interpolate.interp1d
    #data3d_interp=np.array(data3d_keep)
    #data3d_interp[np.isnan(data3d_interp)] = griddata(
    #        (x[~np.isnan(data3d_keep)], y[~np.isnan(data3d_keep)]), # points we know
    #        data3d_keep[~np.isnan(data3d_keep)],                    # values we know
    #        (x[np.isnan(data3d_keep)], y[np.isnan(data3d_keep)]))
#    return data3d_keep
    return data3d_keep





def reject_outliers_img_array(data3d,threshold=3.):
    median_per_img=np.nanmedian(data3d,axis=(1,2))
    n=median_per_img.shape[0]
    median_value=np.nanmedian(median_per_img)
    #print(median_value)
    xx,yy,zz=data3d.shape
    data3d_keep=np.empty((xx,yy,zz))
    data3d_keep[:]=np.nan
    diff=abs(median_per_img-median_value)
    med_diff=np.median(diff)
    for i in range(n):
        if med_diff == 0:
            s=0
        else:
            s=diff[i]/med_diff
            #print(s)
        print(s)
        if s<threshold:
            print(s,threshold)
            data3d_keep[i]=data3d[i]
#        data3d_keep[i]=np.where(s<threshold,data3d[i])
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data3d_keep



def reject_outliers_per_img(data2d,threshold=3.):
    median_value=np.median(np.nanmean(data2d))
    #print(median_value)
    diff=abs(data2d-median_value)
    #print(diff)
    med_diff=np.median(diff)
    #print(med_diff)
#    print('...remove outlier...')
    if med_diff == 0:
        s=0
    else:
        s=diff/med_diff
        #print(s)
    data2d_keep=np.where(s<threshold,data2d,np.NaN)
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data2d_keep



print('-------------------------------')
img1=[np.random.randint(10,size=(3,5,5))]
print(img1)

img1[0][2][0][3]=110 
img1[0][1][3][3]=120 
img1[0][0][3][2]=150 
img1[0][0][2][4]=100   

print(img1)

img1_keep=reject_outliers_2x2img(img1,3)
print(img1_keep)

print('-------------------------------')
img2=[np.random.randint(10,size=(3,5,5))*(0.1)]
print(img2)

img2[0][2][0][3]=np.nan
img2[0][1][3][3]=np.nan
img2[0][0][3][2]=np.nan
img2[0][0][2][4]=np.nan

print(img2)

img2_keep=reject_outliers_2x2img(img2,3)
print(img2_keep)


print('-------------------------------')
img3=[np.random.randint(10,size=(5,4,4))]
img3[0][1]=img3[0][1]*2
img3[0][3]=img3[0][1]*10
print(img3)

img3_keep=reject_outliers_2x2img(img3,3)
print(img3_keep)

print('-------------------------------')
a=np.arange(36.).reshape(4,3,3)
a[3]=a[3]*10

a_keep=reject_outliers_img_array(a)
print(a_keep)