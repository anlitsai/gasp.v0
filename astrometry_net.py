

#import os, sys, shutil, pandas as pd, numpy as np, smtplib, requests, time
import os, sys, shutil
#import pandas as pd 
#import numpy as np
import smtplib
import requests, time
from six.moves import urllib
#from datetime import datetime, timedelta
from glob import glob
#from astropy.io import fits
#from astropy.stats import sigma_clipped_stats
#from photutils import make_source_mask, DAOStarFinder, CircularAperture, aperture_photometry
#from astropy.time import Time
from astropy.wcs import WCS
#from astroML.crossmatch import crossmatch_angular
from selenium import webdriver
from bs4 import BeautifulSoup

files=['/home/altsai/gasp/data/201901/slt20190111/wchen/wchen_03_GASP_01/Mkn501-20190111@211529-R_Astrodon_2018.fts', '/home/altsai/gasp/data/201804/slt20180409/wchen/wchen_03_GASP_01/OJ49-20180409@125922-R_30508.fts']
#files=['/home/altsai/gasp/data/201901/slt20190111/wchen/wchen_03_GASP_01/Mkn501-20190111@211529-R_Astrodon_2018.fts']

print(files)
print(type(files))

# Find astrometric solution of the failed images by online-version Astrometry.net.
#newOfflinePaths = glob("/LWTanaly/{}/neo_red/*.new".format(yesterday))
#redFailPaths = [i for i in redPaths if (i.split('.')[0]+'.new') not in newOfflinePaths]
#driver = webdriver.Chrome()
driver = webdriver.Firefox('/usr/bin/')

driver.implicitly_wait(5)
#for path in redFailPaths:
for path in files:
    driver.get("http://nova.astrometry.net/upload")
    driver.find_element_by_id("id_file").send_keys(path)
    driver.find_element_by_name("submit").submit()
    time.sleep(60)
    res = requests.get(driver.current_url)
    soup = BeautifulSoup(res.text, 'lxml')
    res_calib = requests.get("http://nova.astrometry.net" + soup.select("table")[2].select("a")[1]["href"])
    soup_calib = BeautifulSoup(res_calib.text, 'lxml')
    files = soup_calib.find(id="calibration_table").select("a")
    job = soup_calib.find(id="user_image")["src"].split("/")[-1]
    urllib.request.urlretrieve("http://nova.astrometry.net"+files[1]["href"], path.split('.')[0]+'.new')
driver.quit()



