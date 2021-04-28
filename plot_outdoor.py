# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:01:37 2021

@author: Devineni
"""
import pandas as pd
import numpy as np
import statistics
from statistics import mean
import time
import datetime as dt
import matplotlib.pyplot as plt
import operator # for plotting

from openpyxl import load_workbook

# import mysql.connector
import os
import pymysql
from sqlalchemy import create_engine

from easygui import *
import sys

#from recalibration import clean_sql_reg   

def prRed(skk): print("\033[31;1;m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[33;1;m {}\033[00m" .format(skk)) 
engine = create_engine("mysql+pymysql://root:Password123@localhost/",pool_pre_ping=True)
import datetime
import matplotlib.dates as mdates

import matplotlib.units as munits

#%%
def plot_outdoor(t0, tn):
    engine = create_engine("mysql+pymysql://root:Password123@localhost/weather",pool_pre_ping=True)
    result = pd.read_sql_query("SELECT * FROM {}.{} WHERE datetime BETWEEN '{}' AND \
                        '{}'".format("eshl_winter2", "2d", t0,\
                            tn), con = engine)

    
    result = result.loc[:,['datetime', 'temp_°C', 'RH_%rH', 'CO2_ppm']]
    result = result.set_index("datetime")
    result = result.rolling(120).mean()
    print(result)
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    
    lims = [(np.datetime64('2005-02'), np.datetime64('2005-04')),
        (np.datetime64('2005-02-03'), np.datetime64('2005-02-15')),
        (np.datetime64('2020-02-27 01:00'), np.datetime64('2020-02-27 09:00'))]
    
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    

    
    
    
    par1 = host.twinx()
    par2 = host.twinx()
    
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)
    
    p1, = host.plot(result.index, result['temp_°C'], "b-", label="Temperature (°C)", linewidth=1)
    p2, = par1.plot(result.index, result['CO2_ppm'], "r--", label="CO2 (ppm)", linewidth=1)
    p3, = par2.plot(result.index, result['RH_%rH'], "g-.", label="RH (%)", linewidth=1)
    
    # host.set_xlim(0, 2)
    host.set_ylim(0, 30)
    par1.set_ylim(0, 3000)
    par2.set_ylim(0, 100)
    
    host.set_xlabel("Time")
    host.set_ylabel("Temperature (°C)")
    par1.set_ylabel(r'$\mathrm{CO_2 (ppm)} $')
    par2.set_ylabel("RH (%)")
    
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    
    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=3, maxticks=11)
    formatter = mdates.ConciseDateFormatter(locator)
    host.xaxis.set_major_locator(locator)
    host.xaxis.set_major_formatter(formatter)
    host.set_xlim(lims[2])
    
    lines = [p1, p2, p3]
    
    # plt.title(title)
    
    host.legend(lines, [l.get_label() for l in lines])
    
    plt.savefig('plot.png', bbox_inches='tight', dpi=400)

    plt.show()
    
    
#%%

plot_outdoor('2020-02-27 01:00:00', '2020-02-27 09:00:00')
