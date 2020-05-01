#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:24:27 2019

@author: altsai
"""

import os
import sys
import numpy as np
#import csv
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import datetime
from astropy.io import fits


print('-------------------------------------')
print('Update the fitsheader table for each target')
print('-------------------------------------')

