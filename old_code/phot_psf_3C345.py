#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:16:02 2019

@author: altsai
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils import FittableImageModel
from photutils import prepare_psf_model
from astropy.modeling.functional_models import Moffat2D
#mo=Moffat2D(1,1028,1021)
#print(mo)
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.psf import BasicPSFPhotometry




from astropy.stats import gaussian_sigma_to_fwhm
# print(gaussian_sigma_to_fwhm)
# gaussian_sigma_to_fwhm=2.3548200450309493
sigma_psf = 2.0
fwhm=sigma_psf*gaussian_sigma_to_fwhm
#sigma=fwhm/gaussian_sigma_to_fwhm
# sigma=fwhm/(2.0*np.sqrt(2.0*np.log10(2.0)))????


# group_maker
from photutils.psf import DAOGroup
daogroup = DAOGroup(2.0*fwhm)

fitsfile='3C345-20190714@135841-R_Astrodon_2018_calib.fits'
hdu=fits.open(fitsfile)
imhead=hdu[0].header
imdata=hdu[0].data

# bkg_estimator
from photutils import MMMBackground
bkg_estimator = MMMBackground()
from photutils.background import MADStdBackgroundRMS
bkgrms = MADStdBackgroundRMS()
std = bkgrms(imdata)  # error
threshold=3.5*std


# background level and error
from astropy.stats import sigma_clipped_stats
mean,median,std=sigma_clipped_stats(imdata,sigma=3.)
# print(mean,median,std)
# 1176.6646745343921 1176.0827349796416 39.1334108277639

# fitter
from astropy.modeling.fitting import LevMarLSQFitter
fitter = LevMarLSQFitter()

# fitshape
# fitshape=(5,5) # must be odd values
# fitshape=(7,7) # must be odd values
# fitshape=(9,9) # must be odd values
# 2xfwhm ?


# array shape
xyshape=np.shape(imdata)
n_px=xyshape[0]
#y, x = np.mgrid[:xyshape[0], :xyshape[1]]
x, y = np.mgrid[:n_px, :n_px]

x0=1028.
y0=1021.
model_data=Moffat2D(1,1028,1021)



# psf_model
from photutils.psf import IntegratedGaussianPRF
psf_model = IntegratedGaussianPRF(sigma=sigma_psf)





# 
from astropy.modeling import Fittable2DModel
model2d=Fittable2DModel(model_data)
#model2d = FittableImageModel(model_data)
result = fitter(model_data, x, y, imdata)
#result = fitter(model2d, x, y, imdata)
print(result)
print(np.median(imdata), np.max(imdata), np.sum(imdata))





# finder
# finder receives as input a 2D image and returns an Table object
# which contains columns with names: id, xcentroid, ycentroid, and flux
# In which id is an integer-valued column starting from 1


#
# mask
# A boolean mask with the same shape as data, where 
# a True value indicates the corresponding element of data is masked. 
# Masked pixels are ignored when searching for stars.


mask_imdata=np.full((n_px,n_px),True,dtype=bool)
radius=fwhm*3
#mask0=(x0-radius)**2+(y0-radius)**2

#if

dist_from_center = np.sqrt((x - x0)**2 + (y - y0)**2)

#dist_from_center = (x - n_px)**2 + (y - n_px)**2

#radius = (n_px/2)**2

circular_mask = (dist_from_center > radius)
plt.imshow(circular_mask)

sys.exit(0)

# Building an effective Point Spread Function (ePSF)
# https://photutils.readthedocs.io/en/stable/epsf.html#build-epsf

# PSF Photometry (photutils.psf)
# https://photutils.readthedocs.io/en/stable/psf.html

# create an IterativelySubtractedPSFPhotometry object
# Basic Usage
from photutils.psf import IterativelySubtractedPSFPhotometry
'''
my_photometry = IterativelySubtractedPSFPhotometry(
        finder=my_finder, group_maker=my_group_maker,
        bkg_estimator=my_bkg_estimator, psf_model=my_psf_model,
        fitter=my_fitter, niters=3, fitshape=(7,7))
# get photometry results
photometry_results = my_photometry(image=my_image)
# get residual image
residual_image = my_photometry.get_residual_image()
'''

# Performing PSF Photometry¶
import numpy as np
from astropy.table import Table
from photutils.datasets import (make_random_gaussians_table,
                                make_noise_image,
                                make_gaussian_sources_image)
sigma_psf = 2.0
sources = Table()
sources['flux'] = [700, 800, 700, 800]
sources['x_mean'] = [12, 17, 12, 17]
sources['y_mean'] = [15, 15, 20, 20]
sources['x_stddev'] = sigma_psf*np.ones(4)
sources['y_stddev'] = sources['x_stddev']
sources['theta'] = [0, 0, 0, 0]
sources['id'] = [1, 2, 3, 4]
tshape = (32, 32)
imdata = (make_gaussian_sources_image(tshape, sources) +         
         make_noise_image(tshape, distribution='poisson', mean=6.,                          
                          random_state=1) +
         make_noise_image(tshape, distribution='gaussian', mean=0.,
                          stddev=2., random_state=1))



#from astropy.io import fits
#fitsfile='3C345-20190714@135841-R_Astrodon_2018_calib.fits'
fitsfile='Simulated data'
#hdu=fits.open(fitsfile)
#imhead=hdu[0].header
#imdata=hdu[0].data


from matplotlib import rcParams
rcParams['font.size'] = 13
import matplotlib.pyplot as plt
plt.imshow(imdata, cmap='viridis', aspect=1, interpolation='nearest',
           origin='lower')  
plt.title(fitsfile)  
plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)  

#sys.exit(0)


from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm


bkgrms = MADStdBackgroundRMS()
std = bkgrms(imdata)
iraffind = IRAFStarFinder(threshold=3.5*std,
                          fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                          minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                          sharplo=0.0, sharphi=2.0)
daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
mmm_bkg = MMMBackground()
fitter = LevMarLSQFitter()
psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
#from photutils.psf import IterativelySubtractedPSFPhotometry
photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                group_maker=daogroup,
                                                bkg_estimator=mmm_bkg,
                                                psf_model=psf_model,
                                                fitter=LevMarLSQFitter(),
                                                niters=1, fitshape=(11,11))
result_tab = photometry(image=imdata)
residual_image = photometry.get_residual_image()

plt.subplot(1, 2, 1)
plt.imshow(imdata, cmap='viridis', aspect=1, interpolation='nearest',
               origin='lower')
plt.title(fitsfile)
plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
plt.subplot(1 ,2, 2)
plt.imshow(residual_image, cmap='viridis', aspect=1,
           interpolation='nearest', origin='lower')
plt.title('Residual Image')
plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
plt.show()


sys.exit(0)



# Performing PSF Photometry with Fixed Centroids
psf_model.x_0.fixed = True
psf_model.y_0.fixed = True
pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],
                                        sources['y_mean']])


from photutils.psf import BasicPSFPhotometry

photometry = BasicPSFPhotometry(group_maker=daogroup,
                                bkg_estimator=mmm_bkg,
                                psf_model=psf_model,
                                fitter=LevMarLSQFitter(),
                                fitshape=(11,11))
result_tab = photometry(image=imdata, init_guesses=pos)
residual_image = photometry.get_residual_image()



plt.subplot(1, 2, 1)
plt.imshow(imdata, cmap='viridis', aspect=1,
           interpolation='nearest', origin='lower')
plt.title('Simulated data')
plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
plt.subplot(1 ,2, 2)
plt.imshow(residual_image, cmap='viridis', aspect=1,
           interpolation='nearest', origin='lower')
plt.title('Residual Image')
plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
plt.show()


# Fitting additional parameters¶
gaussian_prf = IntegratedGaussianPRF()
gaussian_prf.sigma.fixed = False
gaussian_prf.sigma.value = 2.05

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.datasets import (make_random_gaussians_table,
                                make_noise_image,
                                make_gaussian_sources_image)
from photutils.psf import (IterativelySubtractedPSFPhotometry,
                           BasicPSFPhotometry)
from photutils import MMMBackground
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.detection import DAOStarFinder
from photutils.detection import IRAFStarFinder
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter

sources = Table()
sources['flux'] = [10000, 1000]
sources['x_mean'] = [18, 9]
sources['y_mean'] = [17, 21]
sources['x_stddev'] = [2] * 2
sources['y_stddev'] = sources['x_stddev']
sources['theta'] = [0] * 2
tshape = (32, 32)
imdata = (make_gaussian_sources_image(tshape, sources) +
         make_noise_image(tshape, distribution='poisson', mean=6.,
                          random_state=1) +
         make_noise_image(tshape, distribution='gaussian', mean=0.,
                          stddev=2., random_state=1))

vmin, vmax = np.percentile(imdata, [5, 95])
plt.imshow(imdata, cmap='viridis', aspect=1, interpolation='nearest',
           origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.show()


daogroup = DAOGroup(crit_separation=8)
mmm_bkg = MMMBackground()
iraffind = IRAFStarFinder(threshold=2.5*mmm_bkg(imdata), fwhm=4.5)
fitter = LevMarLSQFitter()
gaussian_prf = IntegratedGaussianPRF(sigma=2.05)
gaussian_prf.sigma.fixed = False
itr_phot_obj = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                  group_maker=daogroup,
                                                  bkg_estimator=mmm_bkg,
                                                  psf_model=psf_model,
                                                  fitter=fitter,
                                                  fitshape=(11, 11),
                                                  niters=2)


phot_results = itr_phot_obj(image)
phot_results['id', 'group_id', 'iter_detected', 'x_0', 'y_0', 'flux_0']  

# phot_results['sigma_0', 'sigma_fit', 'x_fit', 'y_fit', 'flux_fit']  
phot_results['x_fit', 'y_fit', 'flux_fit']  


plt.imshow(itr_phot_obj.get_residual_image(), cmap='viridis',
aspect=1, interpolation='nearest', origin='lower') 
plt.show()

