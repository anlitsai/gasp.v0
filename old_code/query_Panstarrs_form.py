#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 02:11:55 2019

@author: altsai
"""


#import os
#import sys
#import numpy as np
#import csv
import pandas as pd


#n_query=100

file_in='APT.tbl'
df=pd.read_csv(file_in,delim_whitespace=True,skiprows=2)[:-1]
ra=df['ApertureRA']
dec=df['ApertureDec']

df2=df[['ApertureRA','ApertureDec','Magnitude']]
df2s=df2.sort_values(by='Magnitude',ascending=True)
print(df2s)

df3=df2s[['ApertureRA','ApertureDec']]
df3=df3.rename(columns={'ApertureRA':'ra','ApertureDec':'dec'})
print(df3)

#sys.exit(0)



file_out='query_Panstarrs.csv'
df3.to_csv(file_out,sep=',',index=False,header=True)


