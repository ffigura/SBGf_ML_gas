"""
Created on Fri Nov 13 15:32:22 2020

@author: felipe
"""

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 8))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='k')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='purple')
    ax[4].plot(logs.PE, logs.Depth, '-', color='darkorange')
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)

    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.04)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((11*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR (API)")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(2))
    ax[0].set_ylabel("Depth (m)")
    ax[1].set_xlabel("ILD_log10 \n(ohm.m)")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI (%)")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND (%)")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE (b/e)")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')

    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([]); ax[5].set_xticklabels([])
    
    f.tight_layout(w_pad=0.2)
         
    return


def make_facies_log_plot_SPE(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(6, 8))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='k')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='purple')
    im=ax[4].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[4])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((11*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR (API)")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[0].set_ylabel("Depth (m)")    
    ax[1].set_xlabel("ILD_log10 \n(ohm.m)")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI (%)")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND (%)")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[4].set_xticklabels([])
    
    f.tight_layout(w_pad=0.2)
    
    return

def compare_facies_plot(logs, compadre, facies_colors):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster1 = np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs[compadre].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(9, 8))
    ax[0].plot(logs.GR, logs.Depth, '-',color='darkgreen')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-',color='darkblue')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='k')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='purple')
    ax[4].plot(logs.PE, logs.Depth, '-', color='darkorange')
    im1 = ax[5].imshow(cluster1, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    im2 = ax[6].imshow(cluster2, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im2, cax=cax)
    cbar.set_label((9*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-2):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR (API)")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(2))
    ax[0].set_ylabel("Depth (m)")
    ax[1].set_xlabel("ILD_log10 \n(ohm.m)")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI (%)")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND (%)")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE (b/e)")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('True \nFacies')
    ax[6].set_xlabel(compadre)
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([]); ax[5].set_xticklabels([])
    ax[6].set_xticklabels([]); ax[6].set_yticklabels([])

    return

def regression(data_aug,data,well_name,clr_rfr,test):

    log = (data_aug[data_aug['Well Name'] == well_name]).copy()
    
    rfr = log.drop(['Well Name', 'Depth','Facies'], axis=1)
    y_pred = clr_rfr.predict(rfr.values)
    
    log1 = (data[data['Well Name'] == well_name]).copy()
    log1['Predicted_PE'] = y_pred
    
    log1 = log1.sort_values(by='Depth')
    log2 = test.sort_values(by='Depth')
    
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 6))
    ax[0].plot(log1.PE, log1.Depth, 'black',label='True')
    ax[0].plot(log1.Predicted_PE, log1.Depth, 'darkorange',linestyle='--',label='Predicted')
    ax[0].set_ylim(log1.Depth.max(),log1.Depth.min())
    ax[0].grid()
    ax[0].set_xlabel("PE (b/e) \nTrain")
    ax[0].set_xlim(log1.PE.min(),log1.PE.max())
    ax[0].set_ylim(log1.Depth.max(),log1.Depth.min())
    ax[0].set_ylabel("Depth")
    
    ax[1].plot(log2.PE, log2.Depth, 'black',label='True')
    ax[1].plot(log2.Predicted_PE, log2.Depth, 'darkorange',linestyle='--',label='Predicted')
    ax[1].set_ylim(log2.Depth.max(),log2.Depth.min())
    ax[1].grid()
    ax[1].set_xlabel("PE (b/e) \nTest")
    ax[1].set_xlim(log2.PE.min(),log2.PE.max())
    ax[1].set_ylim(log2.Depth.max(),log2.Depth.min())
    
    f.legend([log2.PE,log2.Predicted_PE], labels=['True','Predicted'], 
             loc = 'upper center', ncol=5, bbox_to_anchor = (0.05,0.03,1,1))
    
    f.tight_layout(w_pad =1.1)
    
    return

def plot1(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')

    ztop=logs.Depth.min(); zbot=logs.Depth.max()

    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)

    f, ax = plt.subplots(nrows=2, ncols=6, figsize=(8, 13.5))
    ax[0,0].plot(logs.GR, logs.Depth, '-',color='darkgreen')
    ax[0,1].plot(logs.ILD_log10, logs.Depth, '-',color='darkblue')
    ax[0,2].plot(logs.DeltaPHI, logs.Depth, '-', color='k')
    ax[0,3].plot(logs.PHIND, logs.Depth, '-', color='purple')
    ax[0,4].plot(logs.PE, logs.Depth, '-', color='darkorange')
    im=ax[0,5].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)

    divider = make_axes_locatable(ax[0,5])
    cax = divider.append_axes("right", size="20%", pad=0.04)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((8*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')

    for i in range(6):
        for j in range(2):
            if j == 0 and i==5:
                pass
            else:
                ax[j,i].set_ylim(ztop,zbot)
                ax[j,i].invert_yaxis()
                ax[j,i].grid()
                ax[j,i].locator_params(axis='x', nbins=3)
    
    ax[1,0].plot(logs['NM_M'], logs.Depth, '.',color='olive')
    ax[1,1].plot(logs['RELPOS'], logs.Depth, '-',color='deeppink')
    ax[1,3].plot(logs['GR^2'], logs.Depth, '-',color='saddlebrown')
    ax[1,4].plot(logs['GR DeltaPHI'], logs.Depth, '-',color='saddlebrown')
    ax[1,5].plot(logs['GR low'], logs.Depth, '-',color='saddlebrown')

    ax[0,0].set_xlabel("GR (API)")
    ax[0,0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[0,0].xaxis.set_major_locator(plt.MaxNLocator(2))
    ax[0,0].set_ylabel("Depth (m)")
    ax[0,1].set_xlabel("ILD_log10 \n(ohm.m)")
    ax[0,1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[0,2].set_xlabel("DeltaPHI (%)")
    ax[0,2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[0,3].set_xlabel("PHIND (%)")
    ax[0,3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[0,4].set_xlabel("PE (b/e)")
    ax[0,4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[0,5].set_xlabel('Facies')

    ax[1,0].set_ylabel("Depth (m)")    
    ax[1,0].set_xlabel("NM_M")
    ax[1,1].set_xlabel("RELPOS")
    ax[1,3].set_ylabel("Depth (m)")     
    ax[1,3].set_xlabel("$GR^{2}$ ($API^{2}$)")
    ax[1,3].set_xticks([0,50000])
    ax[1,4].set_xlabel("GR x DeltaPHI \n(API.%)")
    ax[1,4].set_xlim(logs['GR DeltaPHI'].min(),logs['GR DeltaPHI'].max())
    ax[1,5].set_xlabel('GR low \n(API)')
    ax[1,5].set_xlim(0,logs['GR low'].max())

    ax[0,1].set_yticklabels([]); ax[0,2].set_yticklabels([]); ax[0,3].set_yticklabels([])
    ax[0,4].set_yticklabels([]); ax[0,5].set_yticklabels([]); ax[0,5].set_xticklabels([])

    ax[1,1].set_yticklabels([]); ax[1,5].set_yticklabels([]); ax[1,3].set_yticklabels([])
    ax[1,4].set_yticklabels([])

    ax[1, 2].remove()

    f.tight_layout(w_pad=0.0001)
         
    return

def plot_figure5(data_aug,data,well_name_plot,clr_rfr,test,result):
    
    fig = plt.figure(constrained_layout=False,figsize=(8,8))

    gs = GridSpec(2, 4, figure=fig)
    #feature importance
    ax1 = fig.add_subplot(gs[0, :2])
    sorted_idx2 = result.importances_mean.argsort()

    ax1 = plt.boxplot(result.importances[sorted_idx2].T,
                    vert=True, labels=np.array(data_aug.columns)[sorted_idx2])
    ax1 = plt.title('a)',loc='left')
    ax1 = plt.xticks(rotation=90)
    ax1 = plt.ylabel("Weights")

    #true log
    ax2 = fig.add_subplot(gs[:, 2])

    well_name_plot = 'NOLAN'

    log = (data_aug[data_aug['Well Name'] == well_name_plot]).copy()

    rfr = log.drop(['Well Name', 'Depth','Facies'], axis=1)
    y_pred = clr_rfr.predict(rfr.values)

    log1 = (data[data['Well Name'] == well_name_plot]).copy()
    log1['Predicted_PE'] = y_pred

    log1 = log1.sort_values(by='Depth')


    ax2.plot(log1.PE, log1.Depth, 'black',label='True')
    ax2.plot(log1.Predicted_PE, log1.Depth, 'darkorange',linestyle='--',label='Predicted')
    ax2.set_ylim(log1.Depth.max(),log1.Depth.min())
    ax2.grid()
    ax2.set_xlabel("PE (b/e) \nTrain")
    ax2.set_xlim(log1.PE.min(),log1.PE.max())
    ax2.set_ylim(log1.Depth.max(),log1.Depth.min())
    ax2.set_ylabel("Depth (m)")
    ax2.legend([log1.PE,log1.Predicted_PE], labels=['True','Predicted'])    
    ax2 = plt.title('b)',loc='left')

    #predicted log
    ax3 = fig.add_subplot(gs[:, 3])

    log2 = test.sort_values(by='Depth')

    ax3.plot(log2.PE, log2.Depth, 'black',label='True')
    ax3.plot(log2.Predicted_PE, log2.Depth, 'darkorange',linestyle='--',label='Predicted')
    ax3.set_ylim(log2.Depth.max(),log2.Depth.min())
    ax3.grid()
    ax3.set_xlabel("PE (b/e) \nTest")
    ax3.set_xlim(log2.PE.min(),log2.PE.max())
    ax3.set_ylim(log2.Depth.max(),log2.Depth.min())
    ax3.set_ylabel("Depth (m)")
    ax3.legend([log2.PE,log2.Predicted_PE], labels=['True','Predicted'])    
    ax3 = plt.title('c)',loc='left')


    plt.tight_layout()

    return
