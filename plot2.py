#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:34:35 2022

@author: zachary
"""
#%%
def brp(mouse_cc,mouse_pt,mouse_frx,lfp_cc,lfp_pt,lfp_frx,ecog_cc,ecog_pt,ecog_frx,ylims,dash):   
    import pandas as pd
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size'] = 12    
    ccs = ['#191970',]
    pts = ['#DC143C',]

    ###############################################################################
    p = pd.DataFrame(mouse_pt)
    p = p.add_prefix('voltage')
    p["burst"] = p.index
    p_bursts=pd.wide_to_long(p, ['voltage'], i="burst", j="time")
    p_bursts['which'] = 'PTA'
    c = pd.DataFrame(mouse_cc)
    c = c.add_prefix('voltage')
    c["burst"] = c.index
    c_bursts=pd.wide_to_long(c, ['voltage'], i="burst", j="time")
    c_bursts['which']='CC'
    fig = plt.figure(1, figsize=(9, 12))
    gs = gridspec.GridSpec(6, 6)
    gs.update(wspace=0.1, hspace=0.1)
    center = 1250
    sp = 1000/np.mean(mouse_frx)
    xs = np.arange(center-5*sp, center+5*sp, sp)
    ms = np.arange(center-3*100, center+4*100, 100)
    ###############################################################################
    xtr = fig.add_subplot(gs[0:1, 0:2])
    xtr.text(0.05, 0.9, "A", fontsize=20, fontweight="bold", va="bottom", ha="left",
                  transform=xtr.transAxes)
    xtr = sns.lineplot(data=c_bursts, x="time", y="voltage",hue="which",
                 err_style="band",
                 ci='sd',
                 estimator=np.mean,
                  legend='brief',
                  palette=ccs)
    mean = xtr.get_lines()[0].get_data()
    m = np.concatenate((np.zeros(int(min(mean[0]))), mean[1], np.zeros(1000)))
    xtr.vlines(xs, ylims[0], m[xs.astype(int)], linestyles='dashed', colors=dash)
    xtr.tick_params(bottom=True,labelbottom=False)
    xtr.set_xticks(ms)
    xtr.set_xlim(min(ms), max(ms))
    xtr.set_ylim(ylims)
    xtr.set_yticks([-0.5,0,0.5])
    xtr.set_yticklabels([r'$-0.5 \mu V$','',r'$0.5 \mu V$'])
    xtr.set_xlabel('')  # , fontsize = 9)
    xtr.set_ylabel('')  # , fontsize = 9)
    xtr.legend([], [], frameon=False)
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    # xtr.vlines(xs, ylims[0], mean[xs.astype(int)], linestyles='dashed', colors='red')
    # for xc in xs:
    #     plt.axvline(x=xc,alpha=0.2,color='grey',linestyle='--')
    
    xtr = fig.add_subplot(gs[1:2, 0:2])
    xtr = sns.lineplot(data=p_bursts, x="time", y="voltage",hue="which",
                 err_style="band",
                 ci='sd',
                 estimator=np.mean,
                 legend='brief',
                 palette=pts)
    #xtr.tick_params(bottom=False,labelbottom=False)
    #ms = np.arange(center-3*100, center+4*100, 100)
    mean = xtr.get_lines()[0].get_data()
    m = np.concatenate((np.zeros(int(min(mean[0]))), mean[1], np.zeros(1000)))
    xtr.vlines(xs, ylims[0], m[xs.astype(int)], linestyles='dashed', colors=dash)
    xtr.set_xticks(ms)
    xtr.set_xticklabels(['','-200','','0','','200',''])
    xtr.set_xlim(min(ms), max(ms))
    xtr.set_ylim(ylims)
    xtr.set_yticks([-0.5,0,0.5])
    xtr.set_yticklabels([r'$-0.5 \mu V$','',r'$0.5 \mu V$'])
    xtr.set_xlabel('Time (ms)')  # , fontsize = 9)
    xtr.set_ylabel('')  # , fontsize = 9)
   
    xtr.legend([], [], frameon=False)
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    # for xc in xs:
    #     plt.axvline(x=xc,alpha=0.8,color='gray',linestyle='--')
    
    # xs.astype(int)
    # xtr.vlines(xs, ylims[0], m[xs.astype(int)], linestyles='dashed', colors='red')
    
    ###############################################################################
    p = pd.DataFrame(lfp_pt)
    p = p.add_prefix('voltage')
    p["burst"] = p.index
    p_bursts=pd.wide_to_long(p, ['voltage'], i="burst", j="time")
    p_bursts['which'] = 'PTA'
    c = pd.DataFrame(lfp_cc)
    c = c.add_prefix('voltage')
    c["burst"] = c.index
    c_bursts=pd.wide_to_long(c, ['voltage'], i="burst", j="time")
    c_bursts['which']='CC'
    center = 1250
    sp = 781/np.mean(lfp_frx)
    xs = np.arange(center-7*sp, center+7*sp, sp)
    ###############################################################################
    xtr = fig.add_subplot(gs[0:1, 2:4])
    xtr.text(0.05, 0.9, "B", fontsize=20, fontweight="bold", va="bottom", ha="left",
                  transform=xtr.transAxes)
    xtr = sns.lineplot(data=c_bursts, x="time", y="voltage",hue="which",
                 err_style="band",
                 ci='sd',
                 estimator=np.mean,
                  legend='brief',
                  palette=ccs)
    xtr.tick_params(bottom=True,labelbottom=False)
    xtr.set_xlim(1000, 1500)
    xtr.set_ylim(ylims)

    xtr.set_xlabel('')  # , fontsize = 9)
    xtr.set_yticks([-0.5,0,0.5])   
    xtr.set_yticklabels(['','',''])

    xtr.set_ylabel('')
    xtr.legend([], [], frameon=False)
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    mean = xtr.get_lines()[0].get_data()
    m = np.concatenate((np.zeros(int(min(mean[0]))), mean[1], np.zeros(1000)))
    xtr.vlines(xs, ylims[0], m[xs.astype(int)], linestyles='dashed', colors=dash)
    
    xtr = fig.add_subplot(gs[1:2, 2:4])
    xtr = sns.lineplot(data=p_bursts, x="time", y="voltage",hue="which",
                 err_style="band",
                 ci='sd',
                 estimator=np.mean,
                 legend='brief',
                 palette=pts)
    #xtr.tick_params(bottom=False,labelbottom=False)
    ms = np.arange(center-4*78, center+4*78, 78)
    xtr.set_xticks(ms)
    xtr.set_xticklabels(['','','-200','','0','','200',''])
    xtr.set_xlim(1000, 1500)
    xtr.set_ylim(ylims)


    xtr.set_xlabel('')  # , fontsize = 9)
    xtr.set_ylabel('')
    xtr.set_yticks([])
    xtr.legend([], [], frameon=False)
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    mean = xtr.get_lines()[0].get_data()
    m = np.concatenate((np.zeros(int(min(mean[0]))), mean[1], np.zeros(1000)))
    xtr.vlines(xs, ylims[0], m[xs.astype(int)], linestyles='dashed', colors=dash)
        ###############################################################################
    p = pd.DataFrame(ecog_pt)
    p = p.add_prefix('voltage')
    p["burst"] = p.index
    p_bursts=pd.wide_to_long(p, ['voltage'], i="burst", j="time")
    p_bursts['which'] = 'PTA'
    c = pd.DataFrame(ecog_cc)
    c = c.add_prefix('voltage')
    c["burst"] = c.index
    c_bursts=pd.wide_to_long(c, ['voltage'], i="burst", j="time")
    c_bursts['which']='CC'
    center = 1250
    sp = 781/np.mean(ecog_frx)
    xs = np.arange(center-7*sp, center+7*sp, sp)
    ###############################################################################
    xtr = fig.add_subplot(gs[0:1, 4:6])
    xtr.text(0.05, 0.9, "C", fontsize=20, fontweight="bold", va="bottom", ha="left",
                  transform=xtr.transAxes)
    xtr.text(1, 0.4, "CC", fontsize=16, va="bottom", ha="left",
              transform=xtr.transAxes)
    xtr = sns.lineplot(data=c_bursts, x="time", y="voltage",hue="which",
                 err_style="band",
                 ci='sd',
                 estimator=np.mean,
                  legend='brief',
                  palette=ccs)
    xtr.tick_params(bottom=True,labelbottom=False)
    xtr.set_xlim(1000, 1500)
    xtr.set_ylim(ylims)

    xtr.set_xlabel('')  # , fontsize = 9)
    xtr.set_ylabel('')    
    xtr.set_yticks([-0.5,0,0.5])   
    xtr.set_yticklabels(['','',''])
    xtr.legend([], [], frameon=False)
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    mean = xtr.get_lines()[0].get_data()
    m = np.concatenate((np.zeros(int(min(mean[0]))), mean[1], np.zeros(1000)))
    xtr.vlines(xs, ylims[0], m[xs.astype(int)], linestyles='dashed', colors=dash)
    
    xtr = fig.add_subplot(gs[1:2, 4:6])
    xtr.text(1, 0.4, "PTA", fontsize=16, va="bottom", ha="left",
                  transform=xtr.transAxes)
    xtr = sns.lineplot(data=p_bursts, x="time", y="voltage",hue="which",
                 err_style="band",
                 ci='sd',
                 estimator=np.mean,
                 legend='brief',
                 palette=pts)
    ms = np.arange(center-4*78, center+4*78, 78)
    xtr.set_xticks(ms)
    xtr.set_xticklabels(['','','-200','','0','','200',''])
    xtr.set_xlim(1000, 1500)
    xtr.set_ylim(ylims)
    xtr.set_xlabel('')  # , fontsize = 9)
    xtr.set_ylabel('')   
    xtr.set_yticks([])
    xtr.legend([], [], frameon=False)
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    mean = xtr.get_lines()[0].get_data()
    m = np.concatenate((np.zeros(int(min(mean[0]))), mean[1], np.zeros(1000)))
    xtr.vlines(xs, ylims[0], m[xs.astype(int)], linestyles='dashed', colors=dash)
    plt.show()
    #fig.savefig('plot2.png', format='png', dpi=1200, bbox_inches='tight')
    #fig.savefig('plot2.pdf', format='pdf', dpi=1200, bbox_inches='tight')
#%%
def find_burst_data(c,phase):
    import numpy as np
    anchor = 1250
    cc= []
    pt = []
    for i, burst in c.iterrows():
        b = burst['burst_data']
        p = burst['burst_phase']
        
        if phase:
            first = np.argmin(p)
        else:
            first = np.argmax(np.abs(b))
            if b[first] < 0:
                b = b*-1
        # if any(p > 2.9):    
        #     first = np.argmax(p>2.9)
        b_pad = np.empty((2500))
        b_pad[:] = np.nan
        b_pad[anchor-first:(anchor-first+len(b))] = b  
        if burst['which'] =='cc':
            cc.append(b_pad)
        elif burst['which'] == 'pta':
            pt.append(b_pad)
            
    cc = np.array(cc)
    pt = np.array(pt)
    return cc, pt
#%%
def find_burst_data_mean_session(c,phase):
    import numpy as np
    anchor = 1250
    locs_uniq = c['loc'].unique()
    
    anchor = 1250
    cc_loc = []
    pt_loc = []
    
    for l in locs_uniq:
        loc = c[c['loc']==l]
        cc= []
        pt = []
        for i, burst in loc.iterrows():
            b = burst['burst_data']
            p = burst['burst_phase']
            
            if phase:
                first = np.argmin(p)
            else:
                first = np.argmax(np.abs(b))
                if b[first] < 0:
                    b = b*-1
            b_pad = np.empty((2500))
            b_pad[:] = np.nan
            b_pad[anchor-first:(anchor-first+len(b))] = b  
            if burst['which'] =='cc':
                cc.append(b_pad)
            elif burst['which'] == 'pta':
                pt.append(b_pad)
        cc_mean = np.nanmean(np.array(cc),axis=0)
        pt_mean = np.nanmean(np.array(pt),axis=0)
        cc_loc.append(cc_mean)
        pt_loc.append(pt_mean)
        
    cc = np.array(cc_loc)
    pt = np.array(pt_loc)
    return cc, pt
#%%
def find_burst_data3(c,c1,phase):
    import numpy as np
    anchor = 1250
    locs_uniq = c['loc'].unique()
    
    anchor = 1250
    cc_loc = []
    pt_loc = []
    
    for l in locs_uniq:
        loc = c[c['loc']==l]
        cc= []
        pt = []
        for i, burst in loc.iterrows():
            b = burst['burst_data']
            p = burst['burst_phase']
            
            if phase:
                first = np.argmin(p)
            else:
                first = np.argmax(np.abs(b))
                if b[first] < 0:
                    b = b*-1
            b_pad = np.empty((2500))
            b_pad[:] = np.nan
            b_pad[anchor-first:(anchor-first+len(b))] = b  
            if burst['which'] =='cc':
                cc.append(b_pad)
            elif burst['which'] == 'pta':
                pt.append(b_pad)
        cc_mean = np.nanmean(np.array(cc),axis=0)
        pt_mean = np.nanmean(np.array(pt),axis=0)
        cc_loc.append(cc_mean)
        pt_loc.append(pt_mean)
        
    cc0 = np.nanmean(np.array(cc_loc),axis=0)
    pt0 = np.nanmean(np.array(pt_loc),axis=0)
    
    cc_loc = []
    pt_loc = []
    locs_uniq = c1['loc'].unique()

    for l in locs_uniq:
        loc = c1[c1['loc']==l]
        cc= []
        pt = []
        for i, burst in loc.iterrows():
            b = burst['burst_data']
            p = burst['burst_phase']
            
            if phase:
                first = np.argmin(p)
            else:
                first = np.argmax(np.abs(b))
                if b[first] < 0:
                    b = b*-1
            b_pad = np.empty((2500))
            b_pad[:] = np.nan
            b_pad[anchor-first:(anchor-first+len(b))] = b  
            if burst['which'] =='cc':
                cc.append(b_pad)
            elif burst['which'] == 'pta':
                pt.append(b_pad)
        cc_mean = np.nanmean(np.array(cc),axis=0)
        pt_mean = np.nanmean(np.array(pt),axis=0)
        cc_loc.append(cc_mean)
        pt_loc.append(pt_mean)
        
    cc1 = np.nanmean(np.array(cc_loc),axis=0)
    pt1 = np.nanmean(np.array(pt_loc),axis=0)
    cc = np.array([cc0,cc1])
    pt = np.array([pt0,pt1])

    
    return cc, pt