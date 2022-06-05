#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:03:31 2022

@author: zachary
"""
def sim_plot(mf_pf,mh_ph,xax,zeros,ones,twos,m_f,m_p,n_levels):
    
    from scipy.stats import truncnorm
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    warnings.filterwarnings("ignore")
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size'] = 14
   
    from scipy.stats import lognorm
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size'] = 12
    fig = plt.figure(1, figsize=(9, 12))
    gs = gridspec.GridSpec(7, 5)
    gs.update(wspace=0.7, hspace=0.9)
    
    # false per trial
    xtr = fig.add_subplot(gs[0:2, 0:2])
    xtr.text(-0.5, 1.25, "A", fontsize=20, fontweight="bold", va="bottom", ha="left")
    qs = ['#191970','#FF8247','#DC143C']
    xtr = sns.violinplot(x='sim', y='fal_b_trial', hue='which', inner=None,
                                 data=mf_pf, palette=['0.8','0.8','0.8'], scale="count")
    xtr = sns.stripplot(x='sim', y='fal_b_trial', hue='which',
                                 data=mf_pf, palette=qs, dodge=True)
    xtr.set(xticks=[], xlabel='', ylabel='false / trial')
    xtr.legend([], [], frameon=False)
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_ylim(bottom=0, top=1.25)
    xtr.set(xticks=[])
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25])
    xtr.set_yticklabels(['0','','','', '1', ''])
    xtr.tick_params(axis="y", direction="in")
    xtr.set_xticks([0, 1, 2, 3])
    xtr.set_xticklabels(['Hi', 'Mid', 'Lo', '$1/f$'])
    xtr.tick_params(axis="x", bottom=False)
    # well-recovered
    xtr = fig.add_subplot(gs[2:4, 0:2])
    xtr.text(-0.5, 1, "B", fontsize=20, fontweight="bold", va="bottom", ha="left")
    mh_ph['rec_b_poss'].replace(0, np.nan, inplace=True)
    xtr = sns.violinplot(x='sim', y='rec_b_poss', hue='which', inner=None,
                                 data=mh_ph, palette=['0.8','0.8','0.8'], scale="count")
    xtr = sns.stripplot(x='sim', y='rec_b_poss', hue='which',
                                 data=mh_ph, palette=qs, dodge=True)
    handles, labels = xtr.get_legend_handles_labels()
    l = xtr.legend(handles[3:], labels[3:], bbox_to_anchor=(0.75, 0.5), loc=2, borderaxespad=0.,
                            frameon=False)
    xtr.set(xlabel='', ylabel='well-recovered')
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_ylim(bottom=0, top=1)
    xtr.set_yticks([0, 0.25, 0.5, 0.75, 1])
    xtr.set_yticklabels(['0', '','','', '1'])
    xtr.set_xticks([0, 1, 2, 3])
    xtr.set_xticklabels(['Hi', 'Mid', 'Lo', '$1/f$'])
    xtr.tick_params(axis="x", bottom=False)
    xtr.tick_params(axis="y",direction="in")
    
    # counts 
    xtr = fig.add_subplot(gs[0:1, 2:4])
    barWidth = 0.48
    xtr.text(0, 1.025, "C",fontsize=20, fontweight="bold", va="bottom", ha="left",
                      transform=xtr.transAxes)
    xtr.bar(xax, zeros, color=['0.9'],
                     edgecolor='white', width=barWidth, label="0")
    xtr.bar(xax, ones, bottom=zeros,
                     color=['0.7'], edgecolor='white', width=barWidth, label="1")
    xtr.bar(xax, twos, bottom=[i+j for i, j in zip(zeros, ones)], color=['0.5'],
                     edgecolor='white', width=barWidth, label="2")
    # xtr.bar(xax, threes, bottom=[i+j+k for i, j, k in zip(zeros, ones, twos)],
    #                  color=['0.3'], edgecolor='white', width=barWidth, label='3')
    xtr.set_yticks([0, 0.25, 0.5, .75, 1])
    xtr.set_yticklabels(['0', '25', '50', '75', '100'])
    xtr.set_ylabel('% composition')
    xtr.set_xticks([0, 0.5, 1, 1.5])
    xtr.set_xticklabels(['$Hi$', '$Mid$', '$Lo$', '$1/f$'])
    xtr.legend(loc='lower right', bbox_to_anchor=(
        0.05, -0.1), ncol=1, prop={'size': 8.1}, title='Bursts',frameon=False)
    xtr.spines['right'].set_visible(True)
    xtr.spines['top'].set_visible(False)
    xtr.spines['bottom'].set_visible(False)
    xtr.spines['left'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.yaxis.set_label_position("right")
    xtr.tick_params(axis="x", bottom=False)
    #amplitude
    xtr= fig.add_subplot(gs[1:2,2:3])
    xtr.text(0, 1.025, "D", fontsize=20, fontweight="bold", va="bottom", ha="left",
                      transform=xtr.transAxes)
    
    s = 0.75
    loc = 0.0001
    scale = 0.3
    mean, var, skew, kurt = lognorm.stats(s,loc=loc,scale=scale, moments='mvsk')
    x = np.linspace(lognorm.ppf(0.01, s,loc=loc,scale=scale),
                    lognorm.ppf(0.99, s,loc=loc,scale=scale), 1000)
    xtr.plot(x, lognorm.pdf(x, s,loc=loc,scale=scale),
            'r-', lw=5, alpha=0.6, label='lognorm pdf', color='black')
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set(xticks=[0,1],yticks=[])
    xlabel = xtr.set_xlabel('amplitude', fontsize=12,rotation=-20)
    xtr.xaxis.set_label_coords(0.7, -0.15)
    xtr.set_ylabel('probability', fontsize=10)
    xtr.set_xlim(left=0,right=1.25)
    #frequency
    xtr= fig.add_subplot(gs[1:2,3:4])
    a = 15
    b = 29
    loc = 21
    scale = 1
    x = np.linspace(truncnorm.ppf(0.01, a, b, loc=loc, scale=scale),
                    truncnorm.ppf(0.99, a, b, loc=loc, scale=scale), 100)
    x = np.linspace(15,29,100)
    xtr.plot(x, truncnorm.pdf(x, (a-loc)/scale, (b-loc)/scale, loc=loc, scale=scale),
            'r-', lw=5, alpha=0.6, label='truncnorm pdf', color='black')
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set(xticks=[15,29],yticks=[])
    xtr.set_ylabel('probability', fontsize=10)
    xlabel = xtr.set_xlabel('freq(Hz)', fontsize=12,rotation=-20)
    xtr.xaxis.set_label_coords(0.7, -0.15)
    
    #PSD
    xtr = fig.add_subplot(gs[2:4, 2:4])
    qs = sns.color_palette('Set2', n_levels)
    xtr.text(0, 1, "E", fontsize=20, fontweight="bold", va="bottom", ha="left",
                      transform=xtr.transAxes)
    
    xtr.yaxis.tick_right()
    xtr.yaxis.set_ticks_position('both')
    xtr.yaxis.set_label_position("right")
    label = xtr.set_xlabel('frequency')  # , fontsize = 9)
    xtr.xaxis.set_label_coords(0.5, -0.15)
    xtr.set_ylabel('power($V^2/Hz$)')
    
    zorders = 15, 12, 9, 7, 5, 2, 0
    sizes = 12, 12, 10, 12, 10
    names = ['Hi', 'Mid', 'Lo', '$1/f$']
    markers = ['s','^','o','*']
    mes = [[20],
           [22],
           [18],
           [23]]
    for i, (f, p, z) in enumerate(zip(m_f, m_p, zorders)):
        xtr.loglog(f[0:100], p[0:100], color=qs[i],
                             zorder=z,
                             label=names[i],
                             linewidth = 2)
    # xticks = np.arange(0,501,125)
    xtr.set_ylim(bottom=10e-10)
    xtr.legend(frameon=False,ncol=1,bbox_to_anchor=(
        0.5, 0.65))
    yticks = [10e-6, 10e-3]
    # provide info on tick parameters
    xtr.tick_params(direction='in', which='both', length=8,
                    bottom=True, top=False, left=False, right=True)
    xtr.tick_params(direction='in', which='minor', length=4,
                    bottom=True, top=False, left=False, right=True)
    xtr.tick_params(axis="x", direction="in", pad=2)
    # label = "A"
    # xtr.text(0, 1.5, label, fontsize=15, fontweight="bold", va="bottom", ha="left")
    xtr.set_ylim(bottom=10e-7)
    xtr.set_xlim(left=1,right=100)
    
    plt.yticks(yticks)