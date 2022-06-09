#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:45:28 2022

@author: zachary
"""
def amp_dur(com1,com2,com3):
    
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import numpy as np
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    import matplotlib.pyplot as plt
    import seaborn as sns
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size'] = 12
    fig = plt.figure(1, figsize=(9, 12))
    gs = gridspec.GridSpec(9, 9)
    gs.update(wspace=0.7, hspace=0.9)
    background = 'whitesmoke'
    kde_color= '0.4'
    v_val = 100
    ###############################################################################
    ############################################################################### 
    xtr = fig.add_subplot(gs[0:2, 0:2])
    xtr.text(0, 1.025, "A", fontsize=20, fontweight="bold", va="bottom", ha="left",
                      transform=xtr.transAxes)
    sub = com1[com1['which']=='pta']
    c = sns.color_palette("rocket_r", as_cmap=True)
    xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1,vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])   
    xtr = sns.kdeplot(data=sub, x='duration',y='amp',fill=False,levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='',xlabel='')
    xtr.set_title('PTA')
    xtr.yaxis.set_label_coords(-0.05, 0.5)
    xtr.xaxis.set_label_coords(0.5, -0.05)   
    ###############################################################################
    xtr = fig.add_subplot(gs[0:2, 2:4])
    sub = com1[com1['which']=='cc']
    p = xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1, vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])    
    xtr = sns.kdeplot(data=sub, x='duration',y='amp', levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(xlabel=None,ylabel=None)
    xtr.set_title('CC')    
    ###############################################################################
    xtr = fig.add_subplot(gs[0:2, 4:6])
    sub = com1[com1['which']=='co']
    p = xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1,vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])
    plt.colorbar(p[3],ax=xtr, shrink=0.8, pad = -0.15)
    xtr = sns.kdeplot(data=sub, x='duration',y='amp', levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(xlabel=None,ylabel=None)
    xtr.set_title('CO')
    ##############################################################################
    ##############################################################################
    xtr = fig.add_subplot(gs[2:4, 0:2])
    xtr.text(0, 1.025, "B", fontsize=20, fontweight="bold", va="bottom", ha="left",
                      transform=xtr.transAxes)
    sub = com2[com2['which']=='pta']
    c = sns.color_palette("rocket_r", as_cmap=True)
    xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1,vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])   
    xtr = sns.kdeplot(data=sub, x='duration',y='amp',fill=False,levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='',xlabel='')
    xtr.yaxis.set_label_coords(-0.05, 0.5)
    xtr.xaxis.set_label_coords(0.5, -0.05) 
    ###############################################################################
    xtr = fig.add_subplot(gs[2:4, 2:4])
    sub = com2[com2['which']=='cc']
    #sub['duration'] = sub['duration']/sub['freq']
    p = xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1, vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])    
    xtr = sns.kdeplot(data=sub, x='duration',y='amp', levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(xlabel=None,ylabel=None)
    xtr.set_title('')    
    ###############################################################################
    xtr = fig.add_subplot(gs[2:4, 4:6])
    sub = com2[com2['which']=='co']
    p = xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1,vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])
    xtr = sns.kdeplot(data=sub, x='duration',y='amp', levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(xlabel=None,ylabel=None)
    xtr.set_title('')
    #############################################################################################################
    ############################################################################### 
    xtr = fig.add_subplot(gs[4:6, 0:2])
    xtr.text(0, 1.025, "C", fontsize=20, fontweight="bold", va="bottom", ha="left",
                      transform=xtr.transAxes)
    sub = com3[com3['which']=='pta']
    c = sns.color_palette("rocket_r", as_cmap=True)
    xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1,vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])   
    xtr = sns.kdeplot(data=sub, x='duration',y='amp',fill=False,levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='Amplitude',xlabel='Duration (s)')
    xtr.set_title('')
    xtr.yaxis.set_label_coords(-0.05, 0.5)
    ###############################################################################
    xtr = fig.add_subplot(gs[4:6, 2:4])
    sub = com3[com3['which']=='cc']
    p = xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1, vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])    
    xtr = sns.kdeplot(data=sub, x='duration',y='amp', levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(xlabel=None,ylabel=None)
    xtr.set_title('')
    ###############################################################################
    xtr = fig.add_subplot(gs[4:6, 4:6])
    sub = com3[com3['which']=='co']
    p = xtr.hist2d(sub['duration'],sub['amp'], bins=(25, 25), cmap=c,cmin=0.1,vmin=1, vmax=v_val,
                 range=[[0, 0.5], [0, 1]])
    xtr = sns.kdeplot(data=sub, x='duration',y='amp', levels=5,color=kde_color)
    xtr.set_ylim([0, 1])
    xtr.set_xlim([0, 0.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 0.1, 0.2, 0.3,0.4,0.5])
    xtr.set_xticklabels(['0', '', '', '', '','0.5'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(xlabel=None,ylabel=None)
    xtr.set_title('')
    ################################################################################
    #FREQUENCY
    ################################################################################
    
    qs = ['#191970','#DC143C','#FF8247']
    xtr = fig.add_subplot(gs[0:2, 6:9])
    cc = com1['freq'][com1['which']=='cc']
    pt = com1['freq'][com1['which']=='pta']
    co = com1['freq'][com1['which']=='co']
    xtr.hist([pt,cc,co], 20, histtype='step', stacked=False, fill=False,
                color=[qs[1],qs[0],qs[2]], density=True, label=['PTA','CC','CO'],
                linewidth=2.5)
    xtr.legend()
    xtr.set_ylim([0, 0.5])
    xtr.set_xlim([12, 33])
    xtr.spines['left'].set_visible(True)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.yaxis.set_label_position("right")
    xtr.set_yticks([0, 0.5])
    xtr.set_yticklabels(['0', '0.5'])
    xtr.set_xticks([15, 20, 25, 30])
    xtr.set_xticklabels(['15', '', '', '30'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='',xlabel='')
    xtr.yaxis.set_label_coords(-0.05, 0.5)
    xtr.set_facecolor(background)
    ##########
    xtr = fig.add_subplot(gs[2:4, 6:9])
    cc = com2['freq'][com2['which']=='cc']
    pt = com2['freq'][com2['which']=='pta']
    co = com2['freq'][com2['which']=='co']

    xtr.hist([pt,cc,co], 20, histtype='step', stacked=False, fill=False,
                color=[qs[1],qs[0],qs[2]], density=True, label=['PTA','CC','CO'],
                linewidth=2.5)
    xtr.set_ylim([0, 0.25])
    xtr.set_xlim([12, 33])
    xtr.spines['left'].set_visible(True)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.yaxis.set_label_position("right")
    xtr.set_yticks([0, 0.25])
    xtr.set_yticklabels(['0', '0.25'])
    xtr.set_xticks([15, 20, 25, 30])
    xtr.set_xticklabels(['15', '', '', '30'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='',xlabel='')
    xtr.yaxis.set_label_coords(-0.05, 0.5)
    xtr.set_facecolor(background)
    ##############
    xtr = fig.add_subplot(gs[4:6, 6:9])
    cc = com3['freq'][com3['which']=='cc']
    pt = com3['freq'][com3['which']=='pta']
    co = com3['freq'][com3['which']=='co']

    xtr.hist([pt,cc,co], 20, histtype='step', stacked=False, fill=False,
                color=[qs[1],qs[0],qs[2]], density=True, label=['PTA','CC','co'],
                linewidth=2.5)
    xtr.set_ylim([0, 0.25])
    xtr.set_xlim([12, 33])
    xtr.spines['left'].set_visible(True)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.yaxis.set_label_position("right")
    xtr.set_yticks([0, 0.25])
    xtr.set_yticklabels(['0', '0.25'])
    xtr.set_xticks([15, 20, 25, 30])
    xtr.set_xticklabels(['15', '', '', '30'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='Density',xlabel='Frequency (Hz)')
    xtr.yaxis.set_label_coords(1.05, 0.5)
    xtr.set_facecolor(background)
    plt.show()
    # fig.savefig('/home/zachary/projects/burstmethods/burst_meths/figures/plot1.png', format='png', dpi=1200, bbox_inches='tight')
    # fig.savefig('/home/zachary/projects/burstmethods/burst_meths/figures/plot1.pdf', format='pdf', dpi=1200, bbox_inches='tight')
#%%
def stats(com,measure,dataset):
    from statistics import mean, stdev
    from math import sqrt
    import numpy as np
    print('-------------------------------------------')
    print('Analysis of',dataset, measure)
    if measure == 'count':
        #means = com.groupby(['loc','chan','which'])[measure].mean()
        print(com['which'].value_counts())
    # print(wilcoxon(means.loc[(slice(None),slice(None), 'cc')]-means.loc[(slice(None),slice(None), 'pt')]))
   
    else:    
        means = com.groupby(['loc','chan','which'])[measure].mean()
        shuffles = 10000
        results = []
        a = np.array(means.loc[(slice(None),slice(None), 'cc')])
        b = np.array(means.loc[(slice(None),slice(None), 'pta')])
        cohens_d = (mean(a) - mean(b)) / (sqrt((stdev(a) ** 2 + stdev(b) ** 2) / 2))
        print('CC:', '{:.3}'.format(np.mean(a)), '(', '{:.3}'.format(np.std(a)), ')', 
              'PTA','{:.3}'.format(np.mean(b)), '(', '{:.3}'.format(np.std(b)), ')')
        toShuff = a-b
        results.append(toShuff.mean())
        for i in range(1,shuffles):
            ones = np.random.choice([-1, 1], size=len(toShuff))
            t = toShuff*ones
            results.append(t.mean())
        results = np.array(results)
        
        if results[0] < 0:
            a = np.sum(results[1:]>abs(results[0]))
            b = np.sum(results[1:]<results[0])
        else:
            a = np.sum(results[1:]>results[0])
            b = np.sum(results[1:]<-results[0])
        p = (a+b)/shuffles
        print(str(a+b), 'of', str(shuffles), 'more extreme than observed', 'p= ', str(p))
        print('Cohen\'s d= ', cohens_d)

    print('-------------------------------------------')
    print('')        
    