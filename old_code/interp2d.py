#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 02:15:57 2019

@author: altsai

TIME CONSUMING
"""
import os
import sys
sys.stdout

from astropy.io import fits
import numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from scipy import interpolate


from astropy.utils.data import get_pkg_data_filename
filename = get_pkg_data_filename('/home/altsai/project/20190801.NCU.EDEN/data/gasp/slt201908_master.v1/master_bias_slt201908.fits')

nan=np.nan

'''
data=np.arange(20.).reshape(4,5)
data[3][2]=nan
data[1][4]=nan
data[2][1]=nan
data[3][4]=nan
print(data)
count_nan=np.count_nonzero(np.isnan(data))
print(count_nan)
'''

data=fits.open(filename)[0].data
print(data.shape)

#xn=np.arange(0,data.shape[1])
#yn=np.arange(0,data.shape[0])


np.count_nonzero(~np.isnan(data))
# 4194304=2048x2048
np.count_nonzero(np.isnan(data))
# 0

x_grid = np.arange(0, data.shape[1])
y_grid = np.arange(0, data.shape[0])
#mask invalid values
data2 = np.ma.masked_invalid(data)
xx, yy = np.meshgrid(x_grid, y_grid)
#get only the valid values
x1 = xx[~data2.mask]
y1 = yy[~data2.mask]
newarr = data2[~data2.mask]

print(newarr)

GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
#GD1 = interpolate.interp2d(x1, y1, newarr, kind='linear')
np.count_nonzero(np.isnan(GD1))



hdu=fits.PrimaryHDU(GD1)
hdu.writeto('GD1.fits',overwrite=True) 
#https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python?lq=1
'''
mask=np.isfinite(data)
#mask=np.isnan(data)
z = np.ma.array(data, mask=mask)
x, y = np.mgrid[0:z.shape[0], 0:z.shape[1]]

x1=x[~z.mask]
y1=y[~z.mask]
z1=z[~z.mask]

interp2d(x1,y1,z1)(np.arange(z.shape[0]),np.arange(z.shape[1]))
'''

'''
arr1 = np.array([1, nan, nan, 2, 2, nan, 0])
#Out[90]: array([ 1., nan, nan,  2.,  2., nan,  0.])

arr1_boolean = ~np.isnan(arr1)
#Out[93]: array([ True, False, False,  True,  True, False,  True])

index_arr1_good = arr1_boolean.ravel().nonzero()[0]
#Out[94]: array([0, 3, 4, 6])

arr1_remove_nan = arr1[~np.isnan(arr1)]
#Out[95]: array([1., 2., 2., 0.])

index_arr1_bad = np.isnan(arr1).ravel().nonzero()[0]
#Out[97]: array([1, 2, 5])

arr1[np.isnan(arr1)] = np.interp(index_arr1_bad, index_arr1_good, arr1_remove_nan)
#Out[99]: 
#array([1.        , 1.33333333, 1.66666667, 2.        , 2.        ,
#       1.        , 0.        ])
'''

'''
data_bool=~np.isnan(data)
idx_data_with_value=data_bool.ravel().nonzero()[0]
newdata_remove_nan=data[~np.isnan(data)]
idx_data_with_nan=data[np.isnan(data)]=np.intrp()
'''


'''
xn=data.shape[1]
yn=data.shape[0]

x=np.linspace(0,xn-1,xn)
y=np.linspace(0,yn-1,yn)
#xx,yy=np.meshgrid(x,y)

newdata=interp2d(x,y,data,kind='linear')
#newdata=griddata((x,y),data,(x,y),method='linear')

#data2=scipy.interpolate.RectBivariateSpline(x,y,data,kx=1,ky=1)

plt.plot(newdata)
plot.show()
'''

'''
data=np.ma.masked_invalide(data)

plt.plot(data)
im.show()

xx,yy=np.meshgrid(x,y)
x1=xx[~data.mask]
y1=yy[~data.mask]
newdata=data[~data.mask]

data_interp=griddata((x1,y1),newdata.ravel(),(xx,yy), mothod='linear')
plt.plot(newdata)
im.show()
'''

