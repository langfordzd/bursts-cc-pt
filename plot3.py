#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:29:35 2022

@author: zachary
"""
def behav(l, bs, trials, which, ids):
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)
    import numpy as np
    w = l[(l['which'] == which) | (l['which']=='pt_com')]
    w['behav'] = 'temp'
    w = w.reset_index()
    for i in range(len(w)):#.iterrows():
        r = w.iloc[i]
        t = trials[r['ids']]
        behav = t[9]
        if behav == 1:
            w.at[i, 'behav'] = 'search'
        elif behav == 2:
            w.at[i, 'behav'] = 'repeat'
            
    gr = w.groupby(['loc']).agg({'behav': 'value_counts'})
    gr = gr.rename(columns={'behav':'count'}).reset_index()
    gr = gr.pivot(index=['loc'], columns='behav', values='count')
    gr['ratio_burst'] = gr['search'] / gr['repeat']
    b = pd.DataFrame(bs,columns=['loc','ratio_behav'])
    b = gr.reset_index().merge(b,how='left')
    b['which'] = which

    p = w['ids'].value_counts()
    x = w.groupby(['ids'])['behav'].first()

    ps = pd.DataFrame(p)#.reset_index()
    result = ps.join(x, how='outer').fillna(0)
    result = result.rename(columns={'ids':'count'})
    tot_trials = list(np.arange(0, ids))
    no_bursts = list(set(tot_trials).difference(result.index))
    gs = []
    for n in no_bursts:
        r = w.iloc[i]
        t = trials[r['ids']]
        behav = t[9]
        if behav == 1:
            gs.append([0,'search'])
        elif behav == 2:
            gs.append([0,'repeat'])
    gs = pd.DataFrame(gs,columns = ['count','behav'])
    bh = pd.concat([result, gs], ignore_index=True, axis=0)
    
    
    
    tot_trials = list(np.arange(0, ids))
    zeros = len(list(set(tot_trials).difference(p.index)))
    p = list(p)
    p.extend(np.zeros(zeros))
    temp0 = pd.DataFrame(np.array(p),columns = ['count'])
    temp0['which'] = which
    bh['which'] = which
    
    return b, bh
#%%    
def perm_freq_amp_dur(com, dataset, which, measure):
    import numpy as np
    from statistics import mean, stdev
    from math import sqrt
    means = com.groupby(['loc','which','behav']).mean()

    c = means.loc[(slice(None), which), measure].reset_index()
    a = c[c['behav']==1][measure]
    b = c[c['behav']==2][measure]
    print('--------------------------------------------')
    print('Analysis of', str(dataset), str(measure), 'for', str(which))
    #print(measure)
    #print(which)
    print('search:', '{:.3}'.format(np.mean(a)), '(', '{:.3}'.format(np.std(a)), ')', 
          'repeat','(', '{:.3}'.format(np.mean(b)), ')', '{:.3}'.format(np.std(b)))
    cohens_d = (mean(a) - mean(b)) / (sqrt((stdev(a) ** 2 + stdev(b) ** 2) / 2))
    
    shuffles = 10000
    results = []
    toShuff = np.array(a)-np.array(b)
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
    print('--------------------------------------------')
    print('')
#%%
def perm_binom_test(com,dataset,which):
    import numpy as np
    import pandas as pd
    from scipy import stats

    pd.options.mode.chained_assignment = None 
    shuffles = 10000
    results = []
    w = com[(com['which']==which) | (com['which']=='pt_com')]
    
    toShuff = w['ratio_burst']-w['ratio_behav']
    results.append(toShuff.mean())
    for i in range(1,shuffles):
        ones = np.random.choice([-1, 1], size=len(toShuff))
        t = toShuff*ones
        results.append(t.mean())
    results = np.array(results)
    print('-------------------------------------------')
    print('Analysis of',dataset, which)
    if which == 'cc':
         a = np.sum(results[1:]>results[0])
         p = a/shuffles
         print('Permutation:')
         print(str(a), 'of', str(shuffles), 'more extreme than observed', 'p= ', str(p))
         w['over'] = w['ratio_burst']>w['ratio_behav']
         print('Binomial:')
         print(str(w['over'].sum()), 'out of', str(len(w)),'greater than trial ratio')
         print('p= ', stats.binom_test(w['over'].sum(), n=len(w), p=0.5, alternative='greater'))

    else:
         a = np.sum(results[1:]<results[0])
         p = a/shuffles
         print('Permutation:')
         print(str(a), 'of', str(shuffles), 'more extreme than observed', 'p= ', str(p))
         w['over'] = w['ratio_burst']<w['ratio_behav']
         print('Binomial:')
         print(str(w['over'].sum()), 'out of', str(len(w)), 'less than trial ratio')
         print('p= ', stats.binom_test(len(w)-w['over'].sum(), n=len(w), p=0.5, alternative='less'))
    print('-------------------------------------------')
    print('')
#%%    
def plot_beh(pt_lfp_b,pt_lfp_c,cc_lfp_b,cc_lfp_c,pt_ecog_b,pt_ecog_c,cc_ecog_b,cc_ecog_c,ecog,marco,pablo,v,toSave):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import pandas as pd 
    
    qs = ['#191970','#DC143C','#FF8247']
    count_lfp = pd.concat([pt_lfp_c,cc_lfp_c])
    count_ecog = pd.concat([pt_ecog_c,cc_ecog_c])
    behav_lfp = pd.concat([pt_lfp_b,cc_lfp_b])
    behav_lfp['monkey'] = behav_lfp['loc'].astype(str).str[0]
    behav_ecog = pd.concat([pt_ecog_b,cc_ecog_b])
    
    
    fig = plt.figure(1, figsize=(12, 12))
    gs = gridspec.GridSpec(12, 12)
    gs.update(wspace=0.8, hspace=0.9)
    ################################################################################################
    ############################################################################### 
    xtr = fig.add_subplot(gs[1:2, 0:2])
    xtr.text(0, 1.325, "A", fontsize=20, fontweight="bold", va="bottom", ha="left",
                  transform=xtr.transAxes)
    clfp = count_lfp[count_lfp['behav']=='search']
    xtr = sns.countplot(data = clfp, x='count',hue='which', palette=[qs[1],qs[0]])
    xtr.set_ylim([0, 2500])
    xtr.set_xlim([-0.5, 4.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 2500])
    xtr.set_yticklabels(['0','1500'])
    xtr.set_xticks([0, 1, 2, 3, 4])
    xtr.set_xticklabels(['0', '1', '2','3','4'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='S',xlabel='')
    xtr.yaxis.set_label_coords(-.05, 0.5)
    handles, labels = xtr.get_legend_handles_labels()
    xtr.legend(handles=handles[0:], labels=labels[0:])
    handles, labels = xtr.get_legend_handles_labels()
    xtr.legend(handles=handles[0:],frameon=False, labels=['PTA','CC'],bbox_to_anchor=(1.0, 0.95))
    
    xtr = fig.add_subplot(gs[2:3, 0:2])
    clfp = count_lfp[count_lfp['behav']=='repeat']
    xtr = sns.countplot(data = clfp, x='count',hue='which', palette=[qs[1],qs[0]])
    xtr.set_ylim([0, 1500])
    xtr.set_xlim([-0.5, 4.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 1500])
    xtr.set_yticklabels(['0','1500'])
    xtr.set_xticks([0, 1, 2, 3, 4])
    xtr.set_xticklabels(['0', '1', '2','3','4'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='R',xlabel='# bursts')
    xtr.yaxis.set_label_coords(-.05, 0.5)
    xtr.get_legend().remove()

    xtr = fig.add_subplot(gs[0:3, 2:6])
    palette = {"cc": qs[0], "pt": qs[1]}
    kws = {"s": 60, "facecolor": "none", "linewidth": 1.5}
    xtr = sns.scatterplot(
        data=behav_lfp, x="ratio_behav", y="ratio_burst", 
        edgecolor=behav_lfp["which"].map(palette),#style='monkey',
        **kws,
    )
    handles, labels = zip(*[
        (plt.scatter([], [], ec=color, **kws), key) for key, color in palette.items()
    ])
    xtr.legend(handles=handles[0:], labels=['CC','PTA'], bbox_to_anchor=(0.25, 0.75),frameon=False)
    
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.plot([0, 2], [0, 2], 'k-', lw=1)
    xtr.set_xlim([0,2])
    xtr.set_ylim([0,2])
    xtr.set_yticks([0, 0.5, 1, 1.5, 2])
    xtr.set_yticklabels(['0','','','','2'])
    xtr.set_xticks([0, 0.5, 1, 1.5, 2])
    xtr.set_xticklabels(['0','','','','2'])
    xtr.set(ylabel='',xlabel='')
    xtr.yaxis.set_label_position("right")
    xtr.set_ylabel('S/R bursts')#, fontsize=12)
    xtr.set_xlabel('S/R Trials')#, fontsize=12)
    xtr.xaxis.set_label_coords(0.5, -0.1)   
    xtr.yaxis.set_label_coords(0.925, 0.5)   
    #############################################################################################################################################
    xtr = fig.add_subplot(gs[4:5, 0:2])
    xtr.text(0, 1.325, "B", fontsize=20, fontweight="bold", va="bottom", ha="left",
              transform=xtr.transAxes)
    cecog = count_ecog[count_ecog['behav']=='search']
    xtr = sns.countplot(data = cecog, x='count',hue='which', palette=[qs[1],qs[0]])
    xtr.set_ylim([0, 1500])
    xtr.set_xlim([-0.5, 4.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 1500])
    xtr.set_yticklabels(['0','1500'])
    xtr.set_xticks([0, 1, 2, 3, 4])
    xtr.set_xticklabels(['0', '1', '2','3','4'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='S',xlabel='')
    xtr.xaxis.set_label_coords(0.5, -0.15)
    xtr.yaxis.set_label_coords(-0.05, 0.5)
    xtr.get_legend().remove()
    
    xtr = fig.add_subplot(gs[5:6, 0:2])
    cecog = count_ecog[count_ecog['behav']=='repeat']
    xtr = sns.countplot(data = cecog, x='count',hue='which', palette=[qs[1],qs[0]])
    xtr.set_ylim([0, 7000])
    xtr.set_xlim([-0.5, 4.5])
    xtr.spines['right'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_yticks([0, 1000, 2000, 3000, 4000, 5000, 6000,7000])
    xtr.set_yticklabels(['0','','','','','','','7000'])
    xtr.set_xticks([0, 1, 2, 3, 4])
    xtr.set_xticklabels(['0', '1', '2','3','4'])
    xtr.tick_params(axis="both",direction="in")
    xtr.set(ylabel='R',xlabel='# bursts')
    xtr.yaxis.set_label_coords(-0.05, 0.5)
    xtr.get_legend().remove()
    
    xtr = fig.add_subplot(gs[3:6, 2:6])
    xtr = sns.scatterplot(
        data=behav_ecog, x="ratio_behav", y="ratio_burst", 
        edgecolor=behav_ecog["which"].map(palette),
        **kws,
    )
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.plot([0, 1], [0, 1], 'k-', lw=1)
    xtr.set_xlim([0,1])
    xtr.set_ylim([0,1])
    xtr.set_yticks([0, 0.5, 1, 1])
    xtr.set_yticklabels(['0','','','1'])
    xtr.set_xticks([0, 0.5, 1, 1])
    xtr.set_xticklabels(['0','','','1'])
    xtr.yaxis.set_label_position("right")
    xtr.set_xlabel('S/R Trials')#, fontsize=12)
    xtr.xaxis.set_label_coords(0.5, -0.1)   
    xtr.set_ylabel('S/R bursts')#, fontsize=12)
    xtr.yaxis.set_label_coords(0.925, 0.5)       
    ####################
    ecog_u = ecog[~(ecog['which'] == 'pt_com')]
    lfp_u = pablo[~(pablo['which'] == 'pt_com')]
    marco_u = marco[~(marco['which'] == 'pt_com')]
    lfp_u = lfp_u.astype({'which':object,'amp':float,'freq':float,'duration':float,'behav':object})
    ecog_u = ecog_u.astype({'which':object,'amp':float,'freq':float,'duration':float,'behav':object})
    marco_u = marco_u.astype({'which':object,'amp':float,'freq':float,'duration':float,'behav':object})

    xtr = fig.add_subplot(gs[0:2, 6:8])
    xtr.text(0, 0.7, "C", fontsize=20, fontweight="bold", va="bottom", ha="left",
          transform=xtr.transAxes)
    xtr = sns.violinplot(x="which", y="duration", hue="behav",
                        data=lfp_u, palette=v, split=True,inner="quartile")
    xtr.yaxis.tick_right()
    xtr.set_ylim([0,0.5])
    xtr.set_yticks([0, 0.5])
    xtr.set_yticklabels(['0','0.5'])
    xtr.yaxis.set_label_position("right")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_xlabel('')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])
    xtr.text(0.8, 0.9, "*", fontsize=16, va="bottom", ha="left",
      transform=xtr.transAxes)    


    xtr = fig.add_subplot(gs[2:4, 6:8])
    xtr = sns.violinplot(x="which", y="amp", hue="behav",
                        data=lfp_u, palette=v, split=True,inner="quartile")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.set_ylim([0,1])
    xtr.set_yticks([0, 1])
    xtr.set_yticklabels(['0','1.0'])
    xtr.yaxis.set_label_position("right")
    xtr.set_xlabel('')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])


    xtr = fig.add_subplot(gs[4:6, 6:8])
    xtr = sns.violinplot(x="which", y="freq", hue="behav",
                        data=lfp_u, palette=v, split=True,inner="quartile")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.set_ylim([10,30])
    xtr.set_yticks([10, 30])
    xtr.set_yticklabels(['10','30'])
    xtr.yaxis.set_label_position("right")
    xtr.set_ylabel('')#, fontsize=12)
    xtr.set_xlabel('LFP(P)')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])

    xtr = fig.add_subplot(gs[0:2, 8:10])
    xtr = sns.violinplot(x="which", y="duration", hue="behav",
                        data=marco_u, palette=v, split=True,inner="quartile")
    xtr.yaxis.tick_right()
    xtr.set_ylim([0,0.5])
    xtr.set_yticks([0, 0.5])
    xtr.set_yticklabels(['0','0.5'])
    xtr.yaxis.set_label_position("right")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.set_xlabel('')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])


    xtr = fig.add_subplot(gs[2:4, 8:10])
    xtr = sns.violinplot(x="which", y="amp", hue="behav",
                        data=marco_u, palette=v, split=True,inner="quartile")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.set_ylim([0,1])
    xtr.set_yticks([0, 1])
    xtr.set_yticklabels(['0','1.0'])
    xtr.yaxis.set_label_position("right")
    xtr.set_xlabel('')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])

    xtr = fig.add_subplot(gs[4:6, 8:10])
    xtr = sns.violinplot(x="which", y="freq", hue="behav",
                        data=marco_u, palette=v, split=True, inner="quartile")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.set_ylim([10,30])
    xtr.set_yticks([10, 30])
    xtr.set_yticklabels(['10','30'])
    xtr.yaxis.set_label_position("right")
    xtr.set_ylabel('')#, fontsize=12)
    xtr.set_xlabel('LFP(M)')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])
    xtr.text(0.1, 0.9, "*", fontsize=16, va="bottom", ha="left",
          transform=xtr.transAxes)
    xtr.text(0.8, 0.9, "*", fontsize=16, va="bottom", ha="left",
      transform=xtr.transAxes)

    
    xtr = fig.add_subplot(gs[0:2, 10:12])
    xtr = sns.violinplot(x="which", y="duration", hue="behav",
                        data=ecog_u, palette=v, split=True,inner="quartile")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.set_ylim([0,0.5])
    xtr.set_yticks([0, 0.5])
    xtr.set_yticklabels(['0','0.5'])
    xtr.yaxis.set_label_position("right")
    xtr.set_xlabel('')#, fontsize=12)
    xtr.set_ylabel('Duration')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])
    xtr.text(0.1, 0.9, "*", fontsize=16, va="bottom", ha="left",
      transform=xtr.transAxes)    

    
    xtr = fig.add_subplot(gs[2:4, 10:12])
    xtr = sns.violinplot(x="which", y="amp", hue="behav",
                        data=ecog_u, palette=v, split=True,inner="quartile")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.set_ylim([0,1])
    xtr.set_yticks([0, 1])
    xtr.set_yticklabels(['0','1.0'])
    xtr.yaxis.set_label_position("right")
    xtr.set_xlabel('')#, fontsize=12)
    xtr.set_ylabel('Amplitude')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])
    xtr.text(0.8, 0.9, "*", fontsize=16, va="bottom", ha="left",
      transform=xtr.transAxes)        
    


    xtr = fig.add_subplot(gs[4:6, 10:12])
    xtr = sns.violinplot(x="which", y="freq", hue="behav",
                        data=ecog_u, palette=v, split=True,inner="quartile")
    xtr.spines['left'].set_visible(False)
    xtr.spines['top'].set_visible(False)
    xtr.yaxis.tick_right()
    xtr.set_ylim([10,30])
    xtr.set_yticks([10, 30])
    xtr.set_yticklabels(['10','30'])
    xtr.yaxis.set_label_position("right")
    xtr.set_xlabel('ECoG')#, fontsize=12)
    xtr.set_ylabel('Frequency')#, fontsize=12)
    xtr.get_legend().remove()#%%
    xtr.set_xticklabels(['CC','PTA'])
    handles, labels = xtr.get_legend_handles_labels()
    xtr.legend(handles=handles[0:],frameon=False, labels=['S','R'],bbox_to_anchor=(1.5, -0.1))
    xtr.text(0.8, 0.9, "*", fontsize=16,  va="bottom", ha="left",
      transform=xtr.transAxes)    
    plt.show()
    
    if toSave == True:
        fig.savefig('plot3.png', format='png', dpi=1200, bbox_inches='tight')
        fig.savefig('plot3.pdf', format='pdf', dpi=1200, bbox_inches='tight')

