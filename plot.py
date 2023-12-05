import numpy as np
from ase.io import read,write
import re
import dpdata
import os
from ase import db
import pandas as pd
from ase.db import connect
import matplotlib.pyplot as plt
from matplotlib import ticker

def loss_curve_plot(path,rmse=False,energy=True,force=True,mode='plot',savefig=False,savefigpath=None,show=False,):
    from MLIP_processing.utils import loss_curve
    epoch, rmse_total, rmse_energy, rmse_force = loss_curve(path,energy=energy,force=force)
    fig,ax = plt.subplots()
    if mode=='plot':
        if rmse==True:
            plt.plot(epoch,rmse_total,label='RMSE')
        if energy == True:
            plt.plot(epoch,rmse_energy,label='Energy_RMSE (eV)')
        if force == True:
            plt.plot(epoch,rmse_force,label='Force_RMSE (eV/Å)')
        plt.ticklabel_format(style='plain')

    elif mode=='loglog':
        if rmse==True:
            plt.loglog(epoch,rmse_total,label='RMSE')
        if energy == True:
            plt.loglog(epoch,rmse_energy,label='Energy_RMSE (eV)')
        if force == True:
            plt.loglog(epoch,rmse_force,label='Force_RMSE (eV/Å)')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    elif mode=='semilogx':
        if rmse==True:
            plt.semilogx(epoch,rmse_total,label='RMSE')
        if energy == True:
            plt.semilogx(epoch,rmse_energy,label='Energy_RMSE (eV)')
        if force == True:
            plt.semilogx(epoch,rmse_force,label='Force_RMSE (eV/Å)')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    elif mode=='semilogy':
        if rmse==True:
            plt.semilogy(epoch,rmse_total,label='RMSE')
        if energy == True:
            plt.semilogy(epoch,rmse_energy,label='Energy_RMSE (eV)')
        if force == True:
            plt.semilogy(epoch,rmse_force,label='Force_RMSE (eV/Å)')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    plt.xlabel("Epoch")
    plt.ylabel('RMSE')
    plt.legend()
    
    
    if(savefig==True):
        plt.savefig(savefigpath)
    if(show==True):
        plt.show()