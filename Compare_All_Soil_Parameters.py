# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:32:54 2022

@author: RAY
"""

# =============================================================================
# Impoort
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

import matplotlib.ticker as mticker
import scipy.stats as st
import scipy.optimize as opt
import scipy.io as io
import pickle 
from scipy.optimize import basinhopping
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from scipy import interpolate as interp
import os 
##cutom modules
from fit_class_file import fit_class
from baysian_fit_class_multithread import bayes_fit_class
from linear_shear_class  import linear_shear_class 
from  rotational_shear_data_class  import rotational_shear_data_class 
from query_function import query_prep_shear_objects,query_coulomb_moore
from utility_functions_filter_gradients import filt_cut,filt_cut_raw
from fitstat_functions import r2_fun
from statistical_tests_functions import plot_bayes_test_table,hypothesis_test_is_zero,hypothesis_test_is_zero_morestats
import dill
from  scipy.interpolate import interp1d
import subprocess
import scipy.interpolate as intp
import copy
#%%

pathstr=str(os.path.basename(__file__))
factor_name=pathstr[0:pathstr.find('.')]
factor_name='Compare_all_Parameters'
savefigon=1
# =============================================================================
# Load data
# =============================================================================
######################mohr-coulomb
##Torsion vs linear shear
fitc_129_140=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group1_j'+str(140)+'.obj','rb'))
fitc_300_140=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group2_j'+str(140)+'.obj','rb'))  
fitc_583_140=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group3_j'+str(140)+'.obj','rb'))  

fitc_129_50=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group1_j'+str(50)+'.obj','rb'))
fitc_300_50=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group2_j'+str(50)+'.obj','rb'))  
fitc_583_50=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group3_j'+str(50)+'.obj','rb')) 

fitc_129_250=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group1_j'+str(250)+'.obj','rb'))
fitc_300_250=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group2_j'+str(250)+'.obj','rb'))  
fitc_583_250=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(8)+'_group3_j'+str(250)+'.obj','rb'))  
##Torsion vs linear shear
fitc_300_50=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(10)+'_group1_j'+str(50)+'.obj','rb'))  
fitc_300_50L=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(10)+'_group2_j'+str(50)+'.obj','rb')) 

fitc_300_140=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(10)+'_group1_j'+str(140)+'.obj','rb'))  
fitc_300_140L=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(10)+'_group2_j'+str(140)+'.obj','rb')) 

fitc_300_250=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(10)+'_group1_j'+str(250)+'.obj','rb'))  
fitc_300_250L=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(10)+'_group2_j'+str(250)+'.obj','rb')) 
##shear velocity

##Calulate upper and lower bound as errors for all objects
oblist=[fitc_129_140,fitc_300_140,fitc_583_140,fitc_300_50,fitc_300_50L,fitc_300_250,fitc_300_250L,fitc_129_50,fitc_300_50,fitc_583_50,fitc_129_250,fitc_300_250,fitc_583_250,fitc_300_140,fitc_300_140L]
oblistc=[fitc_129_140,fitc_300_140,fitc_583_140,fitc_300_50,fitc_300_50L,fitc_300_250,fitc_300_250L,fitc_129_50,fitc_300_50,fitc_583_50,fitc_129_250,fitc_300_250,fitc_583_250,fitc_300_140,fitc_300_140L]
# =============================================================================
# for i in range(len(oblist)):
#     oblist[i].LB=np.zeros(len(oblist[i].p2sigma[:,0]))
#     oblist[i].UB=np.zeros(len(oblist[i].p2sigma[:,0]))
#     for j in range(len(oblist[i].p1sigma[:,0])):
#         oblist[i].LB[j]=-oblist[i].p2sigma[j,0]+oblist[i].w_hat[j]
#         oblist[i].UB[j]=oblist[i].p2sigma[j,1]-oblist[i].w_hat[j]
# 
# =============================================================================

# =============================================================================
# for i in range(len(oblist)):
#     oblist[i].LB=np.zeros(len(oblist[i].p1sigma[:,0]))
#     oblist[i].UB=np.zeros(len(oblist[i].p1sigma[:,0]))
#     for j in range(len(oblist[i].p1sigma[:,0])):
#         oblist[i].LB[j]=-oblist[i].p1sigma[j,0]+oblist[i].w_hat[j]
#         oblist[i].UB[j]=oblist[i].p1sigma[j,1]-oblist[i].w_hat[j]
# 
# =============================================================================
if 1==0:
    percentile=97.5#75#84.1
    for i in range(len(oblist)):
        oblist[i].LB=np.zeros(len(oblist[i].p2sigma[:,0]))
        oblist[i].UB=np.zeros(len(oblist[i].p2sigma[:,0]))
        for j in range(len(oblist[i].samples[0,:])):
            oblist[i].LB[j]=-np.percentile(oblist[i].samples[:,j],100-percentile)+oblist[i].w_hat[j]
            oblist[i].UB[j]=np.percentile(oblist[i].samples[:,j],percentile)-oblist[i].w_hat[j]
    sigma_level=2#97.5#75  

#or cov  
if 1==1:
    for i in range(len(oblist)):
        oblist[i].LB=np.zeros(len(oblist[i].p2sigma[:,0]))
        oblist[i].UB=np.zeros(len(oblist[i].p2sigma[:,0]))
        for j in range(len(oblist[i].samples[0,:])):
            oblist[i].LB[j]=np.std(oblist[i].samples[:,j])
            oblist[i].UB[j]=np.std(oblist[i].samples[:,j])
    sigma_level=1#97.5#7

# =============================================================================
# shape parameters
# =============================================================================
##Torsion vs linear shear
fits_129_140=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group1_j'+str(140)+'.obj','rb'))
fits_300_140=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group2_j'+str(140)+'.obj','rb'))  
fits_583_140=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group3_j'+str(140)+'.obj','rb'))  

fits_129_50=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group1_j'+str(50)+'.obj','rb'))
fits_300_50=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group2_j'+str(50)+'.obj','rb'))  
fits_583_50=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group3_j'+str(50)+'.obj','rb')) 

fits_129_250=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group1_j'+str(250)+'.obj','rb'))
fits_300_250=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group2_j'+str(250)+'.obj','rb'))  
fits_583_250=dill.load(open('savedata_'+'Compare_Areas'+'_plt'+str(12)+'_group3_j'+str(250)+'.obj','rb'))  
##Torsion vs linear shear
fits_300_50=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(12)+'_group1_j'+str(50)+'.obj','rb'))  
fits_300_50L=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(12)+'_group2_j'+str(50)+'.obj','rb')) 

fits_300_140=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(12)+'_group1_j'+str(140)+'.obj','rb'))  
fits_300_140L=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(12)+'_group2_j'+str(140)+'.obj','rb')) 

fits_300_250=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(12)+'_group1_j'+str(250)+'.obj','rb'))  
fits_300_250L=dill.load(open('savedata_'+'Compare_lin_vs_torsion'+'_plt'+str(12)+'_group2_j'+str(250)+'.obj','rb')) 
##shear velocity

fits_300_50_F=dill.load(open('savedata_'+'Compare_Shear_Velocity'+'_plt'+str(8)+'_group1_j'+str(50)+'.obj','rb'))  
fits_300_50_S=dill.load(open('savedata_'+'Compare_Shear_Velocity'+'_plt'+str(8)+'_group2_j'+str(50)+'.obj','rb')) 

fits_300_140_F=dill.load(open('savedata_'+'Compare_Shear_Velocity'+'_plt'+str(8)+'_group1_j'+str(140)+'.obj','rb'))  
fits_300_140_S=dill.load(open('savedata_'+'Compare_Shear_Velocity'+'_plt'+str(8)+'_group2_j'+str(140)+'.obj','rb')) 

fits_300_250_F=dill.load(open('savedata_'+'Compare_Shear_Velocity'+'_plt'+str(8)+'_group1_j'+str(250)+'.obj','rb'))  
fits_300_250_S=dill.load(open('savedata_'+'Compare_Shear_Velocity'+'_plt'+str(8)+'_group2_j'+str(250)+'.obj','rb')) 


oblist=[fits_129_140,fits_300_140,fits_583_140,fits_300_50,fits_300_50L,fits_300_250,fits_300_250L,fits_129_50,fits_300_50,fits_583_50,fits_129_250,fits_300_250,fits_583_250,fits_300_140,fits_300_140L,\
        fits_300_50_F,fits_300_50_S,fits_300_140_F,fits_300_140_S,fits_300_250_F,fits_300_250_S]

# =============================================================================
# for i in range(len(oblist)):
#     oblist[i].LB=np.zeros(len(oblist[i].p1sigma[:,0]))
#     oblist[i].UB=np.zeros(len(oblist[i].p1sigma[:,0]))
#     for j in range(len(oblist[i].p1sigma[:,0])):
#         oblist[i].LB[j]=-oblist[i].p1sigma[j,0]+oblist[i].w_hat[j]
#         oblist[i].UB[j]=oblist[i].p1sigma[j,1]-oblist[i].w_hat[j]
# 
# =============================================================================

# =============================================================================
# for i in range(len(oblist)):
#     oblist[i].LB=np.zeros(len(oblist[i].p2sigma[:,0]))
#     oblist[i].UB=np.zeros(len(oblist[i].p2sigma[:,0]))
#     for j in range(len(oblist[i].p1sigma[:,0])):
#         oblist[i].LB[j]=-oblist[i].p2sigma[j,0]+oblist[i].w_hat[j]
#         oblist[i].UB[j]=oblist[i].p2sigma[j,1]-oblist[i].w_hat[j]
# =============================================================================
if 1==0:
    for i in range(len(oblist)):
        oblist[i].LB=np.zeros(len(oblist[i].p2sigma[:,0]))
        oblist[i].UB=np.zeros(len(oblist[i].p2sigma[:,0]))
        for j in range(len(oblist[i].samples[0,:])):
            oblist[i].LB[j]=-np.percentile(oblist[i].samples[:,j],100-percentile)+oblist[i].w_hat[j]
            oblist[i].UB[j]=np.percentile(oblist[i].samples[:,j],percentile)-oblist[i].w_hat[j]
        
errorbarlabel='Interquartile Range (IQR)'#'95% CI'#r'$\pm$ 1 SD' 

if 1==1:
    for i in range(len(oblist)):
        oblist[i].LB=np.zeros(len(oblist[i].p2sigma[:,0]))
        oblist[i].UB=np.zeros(len(oblist[i].p2sigma[:,0]))
        for j in range(len(oblist[i].samples[0,:])):
            oblist[i].LB[j]=np.std(oblist[i].samples[:,j])
            oblist[i].UB[j]=np.std(oblist[i].samples[:,j])
    sigma_level=1#97.5#7       
# =============================================================================
# 
# =============================================================================
plt.close(plt.figure(2))
fig, ax = plt.subplots(1,num=2)
fontsize=12.5
legend_fontsize=fontsize-2.0
fontsize_xlabels=fontsize-2.2
fontsize_ylabels=fontsize-1.5
fontsize_yaxis=fontsize+0.5
labelspacing = 0.2
scalem=1.05
scalem2=1.03
colorfirst='lightgrey'
# =============================================================================
# Plot Friction angle
# =============================================================================
##arrange by cutofff
par=1
Cut50mm = [fitc_129_50.w_hat[par], fitc_300_50.w_hat[par],fitc_583_50.w_hat[par] ,fitc_300_50L.w_hat[par] , 36.1]
Cut50mmLB = [fitc_129_50.LB[par],fitc_300_50.LB[par], fitc_583_50.LB[par], fitc_300_50L.LB[par],0]
Cut50mmUB = [fitc_129_50.UB[par],fitc_300_50.UB[par], fitc_583_50.LB[par], fitc_300_50L.UB[par],0]

Cut140mm = [fitc_129_140.w_hat[par], fitc_300_140.w_hat[par], fitc_583_140.w_hat[par],fitc_300_140L.w_hat[par] ]
Cut140mmLB = [fitc_129_140.LB[par], fitc_300_140.LB[par], fitc_583_140.LB[par], fitc_300_140L.LB[par] ]
Cut140mmUB = [fitc_129_140.UB[par], fitc_300_140.UB[par], fitc_583_140.UB[par], fitc_300_140L.UB[par] ]

Cut250mm = [fitc_129_250.w_hat[par], fitc_300_250.w_hat[par],fitc_300_250L.w_hat[par] ]
Cut250mmLB = [fitc_129_250.LB[par], fitc_300_250.LB[par],fitc_300_250L.LB[par]]
Cut250mmUB = [fitc_129_250.UB[par], fitc_300_250.UB[par], fitc_300_250L.UB[par]]


#color=['C0','C1','C2','C4','C3']
color=['grey','silver','lightgrey','gainsboro','grey']
hatch=[' ', ' ', ' ', '//']
#hatch=['--', '+', 'x', '\\']
#hatch=['*', 'o', 'O', '.']

N = 5
ind = np.arange(N)    # the x locations for the groups
width = 0.25#35         # the width of the bars
ind=np.arange(0,N/5+0.25,0.25)
p1=ax.bar(ind[0:len(ind)-1], Cut50mm[0:len(ind)-1], width, bottom=0, yerr=(Cut50mmLB[0:len(ind)-1],Cut50mmUB[0:len(ind)-1]),hatch=hatch,color=color,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind[0],y= Cut50mm[0],yerr=Cut50mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)

N = 4
ind=np.arange(1.0,1.0+width*N,0.25)
p2 = ax.bar(ind + width, Cut140mm, width, bottom=0, yerr=(Cut140mmLB,Cut140mmUB),hatch=hatch,color=color,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind[0]+ width,y= Cut140mm[0],yerr=Cut140mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)

##Single Error bar for legend
p6=ax.errorbar(x=ind + width,y=Cut140mm,yerr=(Cut140mmLB,Cut140mmUB) ,  fmt=' ',capsize=5,capthick=1.5,color='black',ecolor='black',linewidth=1.5,zorder=100)


N = 3
ind=np.arange(2.0,2.0+width*N,0.25)
p3 = ax.bar(ind + width+ width, Cut250mm, width, bottom=0, yerr=(Cut250mmLB,Cut250mmUB),hatch=[hatch[0],hatch[1],hatch[3]],color=[color[0],color[1],color[3]],error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind[0]+ width+ width,y= Cut250mm[0],yerr=Cut250mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)

N = 4
p4=ax.bar(3.50, Cut50mm[-1], width, bottom=0,color=color[1],hatch=['o'],error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')#C4

ax.set_xticks([0.25+0.25/2,1.5+0.25/2,2.75])
ax.set_xticklabels(['50mm\nCut-Off','140mm\nCut-Off','250mm\nCut-Off'],fontsize=fontsize-1)

#ax.text(-0.25,41,'Same Soil State Different\nBevameter Test Configurations',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='center',horizontalalignment='left')
#ax1.annotate("",xy=(24.5, 0.07), xycoords='data',xytext=(60, 0.26), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1.4))

ax.text(0.25,27,'Best Evidence\n(Linear Shear)',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left')
ax.annotate("",xy=(0.75,27-0.5), xycoords='data',xytext=(0.75, 19.5), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3",lw=1.4))
ax.text(2.6,40,'  Best Evidence\n(Torsional Shear)',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left')
ax.annotate("",xy=(3,40), xycoords='data',xytext=(2.72, 36), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3",lw=1.4))

ax.plot([-0.25*2,+width*3.5],[16.9,16.5],'k--',lw=0.5,zorder=100)
ax.plot([-0.25*2,+width*11.5],[33.1,33.1],'k--',lw=0.5,zorder=100)
ax.plot([-0.25*2,+width*10.5],[37.6,37.6],'k--',lw=0.5,zorder=100)

ax.text(-width*1.5+0.015,16.5+0.2,str(16.5),bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=9,verticalalignment='bottom',horizontalalignment='left')
ax.text(-width*1.5+0.015,33.1+0.15,str(33.1),bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=9,verticalalignment='bottom',horizontalalignment='left')
ax.text(-width*1.5+0.015,37.6+0.15,str(37.6),bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=9,verticalalignment='bottom',horizontalalignment='left')


ax.set_xlim([-1.5*width,3.50+width*1.5])

labels=['Torsional Shear, 129cm$^2$', 'Torsional Shear, 300cm$^2$', r'Torsional Shear, 583cm$^2$','Linear Shear, 300cm$^2$','Shear Box (ASTM 3080-72)',errorbarlabel]#r'$\pm$ 1 SD'
ax.legend([p1[0],p1[1],p1[2],p1[3],p4,p6],labels,loc='upper right',framealpha=1.0,ncol=2,bbox_to_anchor=(0.0, -0.31,1,0.2),prop={'size': legend_fontsize}, mode='expand',labelspacing = labelspacing)
#ax.legend((p1[0],p1[1]), ('50mm Cut-Off', '140mm Cut-Off', '250mm Cut-Off', '250mm Cut-Off', '250mm Cut-Off'),loc='lower right',framealpha=1.0)
plt.ylim([0,45])
ax.set_ylabel(r'Friction Angle $\phi$ [Deg]',fontsize=fontsize_yaxis)
plt.xticks(fontsize=fontsize_xlabels-0.5)
plt.yticks(fontsize=fontsize_ylabels)
#ax._set_xlabel('Friction Angle')
#ax.yaxis.set_units(inch)
#ax.autoscale_view()
plt.tight_layout()
if 1==0:
    fig.set_tight_layout(True)
    plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=900)
    #plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg',format='svg')
    #subprocess.call('inkscape ' +'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg'+' -o '+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.emf',shell=True)
    #os.remove('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg')
    try :
        loc='D:/My Drive/Ray_workn/M\Hand in/Images/'
        plt.savefig(loc+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=300)
        dill.dump(fig,open(loc+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.obj','wb'))        
    except:
        pass
    
errorbarlabel='IQR'#'95% CI'#r'$\pm$ 1 SD'   
#%%
# =============================================================================
# Plot cohesion
# =============================================================================
plt.close(plt.figure(3))
fig, ax = plt.subplots(1,num=3)


##arrange by cutofff
par=0
Cut50mm = [fitc_129_50.w_hat[par], fitc_300_50.w_hat[par],fitc_583_50.w_hat[par] ,fitc_300_50L.w_hat[par] , 0.1]
Cut50mmLB = [fitc_129_50.LB[par],fitc_300_50.LB[par], fitc_583_50.LB[par], fitc_300_50L.LB[par],0]
Cut50mmUB = [fitc_129_50.UB[par],fitc_300_50.UB[par], fitc_583_50.LB[par], fitc_300_50L.UB[par],0]

Cut140mm = [fitc_129_140.w_hat[par], fitc_300_140.w_hat[par], fitc_583_140.w_hat[par],fitc_300_140L.w_hat[par] ]
Cut140mmLB = [fitc_129_140.LB[par], fitc_300_140.LB[par], fitc_583_140.LB[par], fitc_300_140L.LB[par] ]
Cut140mmUB = [fitc_129_140.UB[par], fitc_300_140.UB[par], fitc_583_140.UB[par], fitc_300_140L.UB[par] ]

Cut250mm = [fitc_129_250.w_hat[par], fitc_300_250.w_hat[par],fitc_300_250L.w_hat[par] ]
Cut250mmLB = [fitc_129_250.LB[par], fitc_300_250.LB[par],fitc_300_250L.LB[par]]
Cut250mmUB = [fitc_129_250.UB[par], fitc_300_250.UB[par], fitc_300_250L.UB[par]]

#color=['C0','C1','C2','C4','C3']
#color=['grey','silver','lightgrey','gainsboro','grey']
#hatch=[' ', ' ', ' ', '//']
#hatch=['--', '+', 'x', '\\']
#hatch=['*', 'o', 'O', '.']

N = 5
ind = np.arange(N)    # the x locations for the groups
width = 0.25#35         # the width of the bars
ind=np.arange(0,N/5+0.25,0.25)
p1=ax.bar(ind[0:len(ind)-1], Cut50mm[0:len(ind)-1], width, bottom=0, yerr=(Cut50mmLB[0:len(ind)-1],Cut50mmUB[0:len(ind)-1]),hatch=hatch,color=color,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
p6=ax.errorbar(x=ind[0:len(ind)-1],y=Cut50mm[0:len(ind)-1],yerr=(Cut50mmLB[0:len(ind)-1],Cut50mmUB[0:len(ind)-1]) ,  fmt=' ',capsize=5,capthick=1.5,color='black',ecolor='black',linewidth=1.5,zorder=100)
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind[0],y= Cut50mm[0],yerr=Cut50mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)


N = 4
ind=np.arange(1.0,1.0+width*N,0.25)
p2 = ax.bar(ind + width, Cut140mm, width, bottom=0, yerr=(Cut140mmLB,Cut140mmUB),color=color,hatch=hatch,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind[0]+ width,y= Cut140mm[0],yerr=Cut140mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)

N = 3
ind=np.arange(2.0,2.0+width*N,0.25)
p3 = ax.bar(ind + width+ width, Cut250mm, width, bottom=0, yerr=(Cut250mmLB,Cut250mmUB),hatch=[hatch[0],hatch[1],hatch[3]],color=[color[0],color[1],color[3]],error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind[0]+ width+ width,y= Cut250mm[0],yerr=Cut250mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)

N = 4
p4=ax.bar(3.50, Cut50mm[-1], width, bottom=0,color=color[1],hatch=['o'],error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')#C4

ax.set_xticks([0.25+0.25/2,1.5+0.25/2,2.75])
ax.set_xticklabels(['50mm\nCut-Off','140mm\nCut-Off','250mm\nCut-Off'],fontsize=fontsize-1)

ax.text(3.45,5,'  Shear Box\n'+r'$\mathit{c}\approx$0',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='center')
ax.annotate("",xy=(3.45,0), xycoords='data',xytext=(3.45, 5), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1.4))

#ax.annotate("",xy=(width*10,35), xycoords='data',xytext=(width*11, 35), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1.4),zorder=0)
#ax.text(0.25*11,32,'Small Area Has\nVery High\nUncertainty',bbox=dict(boxstyle='square,pad=.01',facecolor='white', edgecolor='white', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left',zorder=1)

if sigma_level==1:
    height=25              
elif sigma_level==2:
    height=25
elif sigma_level==75:
    height=21.5
ax.annotate("",xy=(width*10,height), xycoords='data',xytext=(width*11, height), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1.4),zorder=0)
ax.text(0.25*11,height,'Small Area Exibits\nHigh Uncertainty',bbox=dict(boxstyle='square,pad=.01',facecolor='white', edgecolor='white', alpha=1),size=11,verticalalignment='center',horizontalalignment='left',zorder=1)



labels=['Torsional Shear, 129cm$^2$', 'Torsional Shear, 300cm$^2$', r'Torsional Shear, 583cm$^2$','Linear Shear, 300cm$^2$','Shear Box (ASTM 3080-72)',errorbarlabel]
#labels=['Torsional Shear, 129cm2 70mm/s', 'Torsional Shear, 300cm2 70mm/s', 'Torsional Shear, 583cm2, 70mm/s','Linear Shear, 300cm2 15mm/s','Shear Box (ASTM 3080-72)']
ax.legend([p1[0],p1[1],p1[2],p1[3],p4,p6],labels,loc='upper right',framealpha=1.0,ncol=2,bbox_to_anchor=(0.0, -0.31,1,0.2),prop={'size': legend_fontsize},mode='expand',labelspacing = labelspacing)
#ax.legend((p1[0],p1[1]), ('50mm Cut-Off', '140mm Cut-Off', '250mm Cut-Off', '250mm Cut-Off', '250mm Cut-Off'),loc='lower right',framealpha=1.0)
if sigma_level==1:
    plt.ylim([0,30])            
elif sigma_level==2:
    plt.ylim([0,40])
elif sigma_level==75:
    plt.ylim([0,25])

ax.set_xlim([-1.5*width,3.50+width*1.5])
ax.set_ylabel(r'Cohesion $\mathit{c}$ [kPa]',fontsize=fontsize_yaxis)
plt.xticks(fontsize=fontsize_xlabels)
plt.yticks(fontsize=fontsize_ylabels)
#ax._set_xlabel('Friction Angle')
#ax.yaxis.set_units(inch)
#ax.autoscale_view()
plt.tight_layout()
#fig.set_tight_layout(True)
if 1==0:
    fig.set_tight_layout(True)
    plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=600)
    #plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg',format='svg')
    #subprocess.call('inkscape ' +'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg'+' -o '+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.emf',shell=True)
    #os.remove('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg')
    try :
        loc='D:/My Drive/Ray_workn/M\Hand in/Images/'
        plt.savefig(loc+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=300)
        dill.dump(fig,open(loc+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.obj','wb'))        
    except:
        pass

#%%
# =============================================================================
# Plot shape parameters
# =============================================================================
##arrange by cutofff
color=['grey','silver','lightgrey','gainsboro','silver']
for par in [0,1,2]:#,3
    
    plt.close(plt.figure(par+4))
    if i==0:
        fig, ax = plt.subplots(1,num=par+4,figsize=[6.4*scalem, 4.8])
    else:
        fig, ax = plt.subplots(1,num=par+4,figsize=[6.4*scalem2, 4.8])
        #fig, ax = plt.subplots(1,num=par+4,figsize=[6.4, 4.8])
    fontsize=12
    hatch=[' ', ' ', ' ', '//','\\']
    
    scale=[1,1,1,1/1000]
    scale_F=[1,1,1,1000]##fast shear
    i=par
    ylabel=[r'Shear Deformation Modulus $\mathit{K}$ [mm]',r'Gradient $\mathit{m}$ [mm$^{-1}$]',r'Scale $\mathit{Y}$ [-]',r'$\sigma_{\epsilon}$ [kPa]']
    
    
    Cut50mm = np.array([fits_129_50.w_hat[par], fits_300_50.w_hat[par],fits_583_50.w_hat[par] ,fits_300_50L.w_hat[par] , fits_300_50_F.w_hat[par+1]*scale_F[i] ])*scale[i]
    Cut50mmLB = np.array([fits_129_50.LB[par],fits_300_50.LB[par], fits_583_50.LB[par], fits_300_50L.LB[par],fits_300_250_F.LB[par+1]*scale_F[i]  ])*scale[i]
    Cut50mmUB = np.array([fits_129_50.UB[par],fits_300_50.UB[par], fits_583_50.LB[par], fits_300_50L.UB[par],fits_300_250_F.UB[par+1]*scale_F[i]  ])*scale[i]
    
    Cut140mm = np.array([fits_129_140.w_hat[par], fits_300_140.w_hat[par], fits_583_140.w_hat[par],fits_300_140.w_hat[par],fits_300_140_F.w_hat[par+1]*scale_F[i]  ])*scale[i]
    Cut140mmLB = np.array([fits_129_140.LB[par], fits_300_140.LB[par], fits_583_140.LB[par], fits_300_140L.LB[par],fits_300_250_F.LB[par+1]*scale_F[i]  ])*scale[i]
    Cut140mmUB = np.array([fits_129_140.UB[par], fits_300_140.UB[par], fits_583_140.UB[par], fits_300_140L.UB[par],fits_300_250_F.UB[par+1]*scale_F[i]  ])*scale[i]
    
    Cut250mm = np.array([fits_129_250.w_hat[par], fits_300_250.w_hat[par],fits_300_250L.w_hat[par],fits_300_250_F.w_hat[par+1]*scale_F[i] ])*scale[i]
    Cut250mmLB = np.array([fits_129_250.LB[par], fits_300_250.LB[par],fits_300_250L.LB[par],fits_300_250_F.LB[par+1]*scale_F[i]])*scale[i]
    Cut250mmUB = np.array([fits_129_250.UB[par], fits_300_250.UB[par], fits_300_250L.UB[par], fits_300_250_F.UB[par+1]*scale_F[i]])*scale[i]
    
    #color=['C0','C1','C2','C4','C3']
    #color=['grey','silver','lightgrey','gainsboro','black']
    #color=['grey','silver','lightgrey','gainsboro','grey']
    #hatch=['--', '+', 'x', '\\']
    #hatch=['*', 'o', 'O', '.']
    
    N = 6
    ind = np.arange(N)    # the x locations for the groups
    width = 0.25#35         # the width of the bars
    ind1=np.arange(0,N*0.25,0.25)
    p1=ax.bar(ind1[0:len(ind1)-1], Cut50mm[0:len(ind)-1], width, bottom=0, yerr=(Cut50mmLB[0:len(ind)-1],Cut50mmUB[0:len(ind)-1]),hatch=hatch,color=color,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
    ##error bar
    p6=ax.errorbar(x=ind1[0:len(ind1)-1],y= Cut50mm[0:len(ind)-1],yerr=(Cut50mmLB[0:len(ind)-1],Cut50mmUB[0:len(ind)-1]) ,  fmt=' ',capsize=5,capthick=1.5,color='black',ecolor='black',linewidth=1.5,zorder=100)
    
    #lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind1[0:len(ind1)-1],y= Cut50mm[0:len(ind)-1],yerr=Cut50mmLB[0:len(ind)-1] ,  fmt=' ',capsize=0,capthick=1.5,color='white',ecolor='white',linewidth=1.5,zorder=101, uplims=[True]*5  )    
    lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind1[0],y= Cut50mm[0],yerr=Cut50mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
    caplines1[0].set_marker('_')
    caplines1[0].set_markersize(10)
    
    N = 5
    ind2=np.arange(ind1[-1],ind1[-1]+width*N,0.25)
    p2 = ax.bar(ind2 + width, Cut140mm, width, bottom=0, yerr=(Cut140mmLB,Cut140mmUB),hatch=hatch,color=color,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
    
    lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind2[0]+ width,y= Cut140mm[0],yerr=Cut140mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
    caplines1[0].set_marker('_')
    caplines1[0].set_markersize(10)
    
    N =4
    ind3=np.arange(ind2[-1]+width,ind2[-1]+width+width*N,0.25)
    p3 = ax.bar(ind3 + width+ width, Cut250mm, width, bottom=0, yerr=(Cut250mmLB,Cut250mmUB),hatch=[hatch[0],hatch[1],hatch[3],hatch[4]],color=[color[0],color[1],color[3],color[4]],error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')

    lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind3[0]+ width+ width,y= Cut250mm[0],yerr=Cut250mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
    caplines1[0].set_marker('_')
    caplines1[0].set_markersize(10)
    #N = 4
    #p4=ax.bar(3.50, Cut50mm[-1], width, bottom=0,color=color[-1],hatch=['\\'],error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')#C4
    
    ax.set_xticks([(ind1[0]+ind1[-1])/2.0-width/2.0,(ind2[0]+ind2[-1])/2.0+ width,(ind3[0]+ind3[-1])/2.0+ width+ width])
    ax.set_xticklabels(['50mm\nCut-Off','140mm\nCut-Off','250mm\nCut-Off'],fontsize=fontsize-2)
    
    #ax.text(-0.25,41,'Same Soil State Different\nBevameter Test Configurations',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='center',horizontalalignment='left')
    #ax1.annotate("",xy=(24.5, 0.07), xycoords='data',xytext=(60, 0.26), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1.4))
    
    #ax.text(0.25,32,'Best Evidence\n(Linear Shear)',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left')
    #ax.annotate("",xy=(0.75,31), xycoords='data',xytext=(0.75, 19.5), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3",lw=1.4))
    #ax.text(2.6,40,'  Best Evidence\n(Torsional Shear)',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left')
    #ax.annotate("",xy=(2.75,40), xycoords='data',xytext=(2.75, 36), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3",lw=1.4))
    
    if i ==0:
        if sigma_level==1:
            ax.set_ylim([0,12])  
            height=7
        elif sigma_level==2:
            ax.set_ylim([0,15])
            height=7
        elif sigma_level==75:
            ax.set_ylim([0,10])
            height=7.5
        ax.text(0.25*2.6,height,'Shear Annulus Size\nSignificantly Affects\nRate of Intial\nStress Increase',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left')
        ax.annotate("",xy=(0.25*2.6+0.25,height), xycoords='data',xytext=(0.25*2.5, 6.5), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3",lw=1.4))
        
    if i ==1:   
        if sigma_level==1:
            ax.set_ylim([0,0.03])            
        elif sigma_level==2:
            ax.set_ylim([0,0.02])
        elif sigma_level==75:
            ax.set_ylim([0,0.01])
            
        ax.text(0.25*9.5,0.0055,'Significantly Higher\nNon-Asymptotic Gradient\nFor Small Area',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left')
        ax.annotate("",xy=(0.25*12,0.0055), xycoords='data',xytext=(0.25*12, 0.0046), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3",lw=1.4))
    if i ==2:
        ax.set_ylim([0,1.03])
        ax.set_yticks(np.arange(0,1.1,0.1))
        ax.annotate("",xy=(0.25*12,0.57), xycoords='data',xytext=(0.25*12, 0.9), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",lw=1.4),zorder=0)
        ax.text(0.25*9.5,0.83,'46% of Maximum Shear\nStress Attributed to Non-\nAsymptotic Gradient',bbox=dict(boxstyle='square,pad=.01',facecolor='white', edgecolor='white', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left',zorder=1)
        
        ax.plot([-0.25*2,+width*11.5],[0.535]*2,'k--',lw=0.5,zorder=100)
        ax.plot([-0.25*2,+width*3.5],[0.895]*2,'k--',lw=0.5,zorder=100)
        ax.text(-width*1.5+0.015,0.54+0.003,str(0.54),bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=9,verticalalignment='bottom',horizontalalignment='left')
        ax.text(-width*1.5+0.015,0.9+0.003,'0.90',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=9,verticalalignment='bottom',horizontalalignment='left')    
   
    ax.set_xlim([-width*1.5,ind3[-1]+width*3.5])

    
    labels=['Torsional Shear, 129cm$^2$', 'Torsional Shear, 300cm$^2$', r'Torsional Shear, 583cm$^2$','Linear Shear, 300cm$^2$','Torsional Shear, 300cm$^2$, 600mm/s',errorbarlabel]
    ax.legend([p1[0],p1[1],p1[2],p1[3],p1[4],p6],labels,loc='upper left',framealpha=1.0,ncol=2,bbox_to_anchor=(0.0, -0.3,1,0.2),prop={'size': legend_fontsize},mode='expand',labelspacing = labelspacing)
    #ax.legend((p1[0],p1[1]), ('50mm Cut-Off', '140mm Cut-Off', '250mm Cut-Off', '250mm Cut-Off', '250mm Cut-Off'),loc='lower right',framealpha=1.0)
    #plt.ylim([0,30])
    ax.set_ylabel(ylabel[i],fontsize=fontsize_yaxis)
    #ax._set_xlabel('Friction Angle')
    #ax.yaxis.set_units(inch)
    #ax.autoscale_view()
    plt.xticks(fontsize=fontsize_xlabels)
    plt.yticks(fontsize=fontsize_ylabels)
    
    plt.tight_layout()
    if 1==0:
        #figcorn.set_tight_layout(True)
        plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=900)
        #plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg',format='svg')
        #subprocess.call('inkscape ' +'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg'+' -o '+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.emf',shell=True)
        #os.remove('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg') 
        try :
            loc='D:/My Drive/Ray_workn/M\Hand in/Images/'
            plt.savefig(loc+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=300)
            dill.dump(fig,open(loc+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.obj','wb'))        
        except:
            pass
# =============================================================================
# 
# =============================================================================
#%%


def coulomb_fun(sigma,w):  
    return w[0]+sigma*np.tan(np.deg2rad(w[1]))

def coulomb_funi(sigma,w):  
    return w[:,0]+sigma*np.tan(np.deg2rad(w[:,1]))

sigma=93#129#93#93#10#93.0 
devidesigma=sigma#1
jmax=250.0
i=0   
percentile=97.5
#percentile=84

for i in range(len(oblistc)):
    oblistc[i].tausamples=np.zeros(len(oblistc[i].p1sigma[:,0]))
    oblistc[i].LBtau=np.zeros(len(oblistc[i].p1sigma[:,0]))
    oblistc[i].UBtau=np.zeros(len(oblistc[i].p1sigma[:,0]))
    
    ##evualuate finction
    mu=coulomb_funi(sigma,oblistc[i].samples[:,0:2])
    oblistc[i].tausamples=(st.norm.rvs(loc=mu,scale=oblistc[i].samples[:,2]))/devidesigma
        
    ##find map
    kde = st.gaussian_kde(oblistc[i].tausamples,0.2) 
    T=np.percentile(oblistc[i].tausamples,0.6)-np.percentile(oblistc[i].tausamples,0.3)       
    w_hat=opt.basinhopping(lambda x: -kde(x),x0=[np.mean(oblistc[i].tausamples)],niter=8,T=T,minimizer_kwargs={'method':'L-BFGS-B','tol':10**-9})['x'][0]
    
    oblistc[i].w_hattau=w_hat
    oblistc[i].LBtau=w_hat-np.percentile(oblistc[i].tausamples, 100-percentile)
    oblistc[i].UBtau=-w_hat+np.percentile(oblistc[i].tausamples, percentile)

##special fast shear fit parameters are different
oblist_F=[fits_300_50_F,fits_300_140_F,fits_300_250_F]
for i in range(len(oblist_F)): 
    oblist_F[i].tausamples=np.zeros(len(oblist_F[i].samples[:,0]))
    oblist_F[i].LBtau=np.zeros(len(oblist_F[i].samples[:,0]))
    oblist_F[i].UBtau=np.zeros(len(oblist_F[i].samples[:,0]))
    
    ##evualuate finction
    oblist_F[i].tausamples=oblist_F[i].samples[:,0]/devidesigma
        
    ##find map
    kde = st.gaussian_kde(oblist_F[i].tausamples,0.2) 
    T=np.percentile(oblist_F[i].tausamples,0.6)-np.percentile(oblist_F[i].tausamples,0.3)       
    w_hat=opt.basinhopping(lambda x: -kde(x),x0=[np.mean(oblist_F[i].tausamples)],niter=8,T=T,minimizer_kwargs={'method':'L-BFGS-B','tol':10**-9})['x'][0]
    
    oblist_F[i].w_hattau=w_hat
    oblist_F[i].LBtau=w_hat-np.percentile(oblist_F[i].tausamples, 100-percentile)
    oblist_F[i].UBtau=-w_hat+np.percentile(oblist_F[i].tausamples, percentile)
    
##Shear Box
Shearbox=coulomb_fun(sigma,[0,36.1])/devidesigma
    
#%%
plt.close(plt.figure(7))
#fig, ax = plt.subplots(1,num=7)
fig, ax = plt.subplots(1,num=7,figsize=[6.4*scalem2, 4.8])

Cut50mm = [fitc_129_50.w_hattau, fitc_300_50.w_hattau,fitc_583_50.w_hattau ,fitc_300_50L.w_hattau,fits_300_50_F.w_hattau ]
Cut50mmLB = [fitc_129_50.LBtau,fitc_300_50.LBtau, fitc_583_50.LBtau, fitc_300_50L.LBtau,fits_300_50_F.LBtau]
Cut50mmUB = [fitc_129_50.UBtau,fitc_300_50.UBtau, fitc_583_50.LBtau, fitc_300_50L.UBtau,fits_300_50_F.UBtau]
color50=['grey','silver','lightgrey','gainsboro','silver']
hatch50=[' ', ' ', ' ', '//','\\']


Cut140mm = [fitc_129_140.w_hattau, fitc_300_140.w_hattau, fitc_583_140.w_hattau,fitc_300_140L.w_hattau,fits_300_140_F.w_hattau ]
Cut140mmLB = [fitc_129_140.LBtau, fitc_300_140.LBtau, fitc_583_140.LBtau, fitc_300_140L.LBtau,fits_300_140_F.LBtau ]
Cut140mmUB = [fitc_129_140.UBtau, fitc_300_140.UBtau, fitc_583_140.UBtau, fitc_300_140L.UBtau,fits_300_140_F.UBtau ]
color140=['grey','silver','lightgrey','gainsboro','silver']
hatch140=[' ', ' ', ' ', '//','\\']


Cut250mm = [fitc_129_250.w_hattau, fitc_300_250.w_hattau,fitc_300_250L.w_hattau,fits_300_250_F.w_hattau ]
Cut250mmLB = [fitc_129_250.LBtau, fitc_300_250.LBtau,fitc_300_250L.LBtau,fits_300_250_F.LBtau]
Cut250mmUB = [fitc_129_250.UBtau, fitc_300_250.UBtau, fitc_300_250L.UBtau,fits_300_250_F.UBtau]
color250=['grey','silver','lightgrey','silver']
hatch250=[' ', ' ', '//','\\']

##plott bars
##50mm
N = len(Cut50mm)
ind = np.arange(N)    # the x locations for the groups
width = 0.25#35         # the width of the bars
ind1=np.arange(0,N*0.25,0.25)
p1=ax.bar(ind1, Cut50mm, width, bottom=0, yerr=(Cut50mmLB,Cut50mmUB),hatch=hatch50,color=color50,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
##errorbar
p6=ax.errorbar(x=ind1,y= Cut50mm,yerr=(Cut50mmLB,Cut50mmUB) ,  fmt=' ',capsize=5,capthick=1.5,color='black',ecolor='black',linewidth=1.5,zorder=100)
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind1[0],y= Cut50mm[0],yerr=Cut50mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)

N = len(Cut140mm)
ind2=np.arange(ind1[-1],ind1[-1]+width*N,0.25)
p2 = ax.bar(ind2 + width+ width, Cut140mm, width, bottom=0, yerr=(Cut140mmLB,Cut140mmUB),hatch=hatch140,color=color140,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind2[0]+ width+ width,y= Cut140mm[0],yerr=Cut140mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)


N =len(Cut250mm)
ind3=np.arange(ind2[-1]+width,ind2[-1]+width+width*N,0.25)
p3 = ax.bar(ind3 + width+ width+ width, Cut250mm, width, bottom=0, yerr=(Cut250mmLB,Cut250mmUB),hatch=hatch250,color=color250,error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')
lotline1, caplines1, barlinecols1 =ax.errorbar(x=ind3[0]+ width+ width+ width,y= Cut250mm[0],yerr=Cut250mmLB[0] ,  fmt=' ',capsize=0,capthick=1.5,color=colorfirst,ecolor=colorfirst,linewidth=1.5,zorder=101, uplims=[True]*1  )
caplines1[0].set_marker('_')
caplines1[0].set_markersize(10)
##sheabox
#p4=ax.bar(ind3[-1] + width+ width+ width+ width+ width, Shear Boxtau, width, bottom=0,color='silver',hatch=['o'],error_kw=dict(lw=1.5, capsize=5, capthick=1.5),edgecolor='black')#C4


ax.set_xticks([(ind1[0]+ind1[-1])/2.0,(ind2[0]+ind2[-1])/2.0+ width*2,(ind3[0]+ind3[-1])/2.0+ width*3])
ax.set_xticklabels(['50mm\nCut-Off','140mm\nCut-Off','250mm\nCut-Off'],fontsize=fontsize-2)

#ax.text(-0.25,41,'Same Soil State Different\nBevameter Test Configurations',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='center',horizontalalignment='left')
#ax1.annotate("",xy=(24.5, 0.07), xycoords='data',xytext=(60, 0.26), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1.4))

ax.text(0.25*3,0.69,'Best Evidence\n(Linear Shear)',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='center')
ax.annotate("",xy=(0.75,0.69), xycoords='data',xytext=(0.75, 0.56), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3",lw=1.4))
ax.text(3.03,0.95,'  Best Evidence\n(Torsional Shear)',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='bottom',horizontalalignment='left')
ax.annotate("",xy=(3.5,0.95), xycoords='data',xytext=(3.25, 0.87), textcoords='data',arrowprops=dict(arrowstyle="<-",connectionstyle="arc3",lw=1.4))


ax.text(-0.25+0.25*1,1.02,'Shear Stress Predicted By Model\nAt '+str(round(sigma,0))+'kPa  & 250mm Displacment',bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=11,verticalalignment='center',horizontalalignment='left')
#ax.annotate("",xy=(24.5, 0.07), xycoords='data',xytext=(60, 0.26), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",lw=1.4))
ax.plot([-0.25+0.25*1,0.25*8.3],[0.96,0.96],'k-',zorder=100)


labels=['Torsional Shear, 129cm$^2$', 'Torsional Shear, 300cm$^2$', r'Torsional Shear, 583cm$^2$','Linear Shear, 300cm$^2$','Torsional Shear, 300cm$^2$, 600mm/s',errorbarlabel]
ax.legend([p1[0],p1[1],p1[2],p1[3],p1[4],p6],labels,loc='upper left',framealpha=1.0,ncol=2,bbox_to_anchor=(0.0, -0.3,1,0.2),prop={'size': legend_fontsize},mode='expand',labelspacing = labelspacing)

ax.plot([-0.25*2,+width*3.5],[0.53]*2,'k--',lw=0.5,zorder=100)
ax.plot([-0.25*2,+width*12.5],[0.824]*2,'k--',lw=0.5,zorder=100)
ax.plot([-0.25*2,+width*11.5],[0.932]*2,'k--',lw=0.5,zorder=100)

ax.text(-width*1.5+0.015,0.93+0.005,str(0.93),bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=9,verticalalignment='bottom',horizontalalignment='left')
ax.text(-width*1.5+0.015,0.82+0.005,str(0.82),bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=9,verticalalignment='bottom',horizontalalignment='left')
ax.text(-width*1.5+0.015,0.53+0.005,str(0.53),bbox=dict(boxstyle='square,pad=.01',facecolor='none', edgecolor='none', alpha=1),size=9,verticalalignment='bottom',horizontalalignment='left')

#ax.plot([-0.25*2,+width*10.5],[37.6,37.6],'k--',lw=0.5,zorder=100)


ax.set_ylim([0,1.1])
ax.set_xlim([-width*1.5,ind3[-1]+width*4.5])
ax.set_yticks(np.arange(0,1.2,0.1))
plt.xticks(fontsize=fontsize-1)
plt.yticks(fontsize=fontsize-1)

#ax.grid(axis='y',zorder=1)
ax.set_ylabel(r'Shear Stress Ratio [$\tau / \sigma$]',fontsize=fontsize_yaxis)
fig.set_tight_layout(True)

#ax2=ax.twinx()
#ax2.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks()) )  )
#ax2.set_yticklabels(np.round(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks()) )*sigma,2))
#ax2.set_ylabel(r'Shear Stress [kPa]',fontsize=fontsize)

#%%
if 1==0:
    from utility_functions_savesvg import convert_to_emf,convert_to_emf_grey
    #plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=1300)  
    #filename='fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png'
    #convert_to_emf(12,speckles=40,smooth=0.75,optimize=1,setingson=True,sharpen=1,filename=filename)
    #import pdb;pdb.set_trace()
    #convert_to_emf_grey(12,speckles=30,smooth=0.3,optimize=0.2,setingson=True,sharpen=0,filename=filename)
    #os.remove('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png') 
        
    #figcorn.set_tight_layout(True)
    plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=900)    
    #plt.savefig('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg',format='svg')    
    #subprocess.call('inkscape ' +'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg'+' -o '+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.emf',shell=True)
    #os.remove('fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.svg') 
        
    try :
        loc='D:/My Drive/Ray_workn/M\Hand in/Images/'
        plt.savefig(loc+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.png',dpi=300)
        dill.dump(fig,open(loc+'fig_'+factor_name+'_plt'+str(plt.gcf().number)+'.obj','wb'))        
    except:
        pass
#%%
# =============================================================================
# calulate R2
# =============================================================================
from baysian_fit_class_multithread_hierachial import bayes_fit_class

oblistc=[fitc_129_140,fitc_300_140,fitc_583_140,fitc_300_50,fitc_300_50L,fitc_300_250,fitc_300_250L,fitc_129_50,fitc_300_50,fitc_583_50,fitc_129_250,fitc_300_250,fitc_583_250,fitc_300_140,fitc_300_140L]


def coulomb_fun(sigma,w):  
    return w[0]+sigma*np.tan(np.deg2rad(w[1]))

def janoshi_local_lin_fun(j,K,taumax):
     result=taumax*Y*(1-np.exp(-j/(K))) 
     return result 

def janoshi_local_fun_lin_unpack(data,yData,c,phi,K):
    j = data
    sigma = yData   
    taumax=coulomb_fun(sigma/1000,[c,phi])*1000
    tau=janoshi_local_lin_fun(j,K,m,Y,taumax)   
    return tau

def wrap_janoshi_local_fun_lin_unpack(data,w,ydata):
    return janoshi_local_fun_lin_unpack(data,ydata,c=w[0],phi=w[1],K=w[2],m=w[3],Y=w[4])

Nparr=3
Nparrall=Nparr+1
#idx=np.round(np.linspace(0,len(xData_group1)-1,200),0).astype(int)
runfit=True
if runfit==False:
    if 1==1: ##seperate regresion
        fitshape_group1=bayes_fit_class(xData_group1,zData_group1,lambda data,w : wrap_janoshi_local_fun_lin_unpack(data,np.hstack([fitc_group1.w_hat[0:2],w]),yData_group1),Nparr,'Linear fit')   
        fitshape_group2=bayes_fit_class(xData_group2,zData_group2,lambda data,w : wrap_janoshi_local_fun_lin_unpack(data,np.hstack([fitc_group2.w_hat[0:2],w]),yData_group2),Nparr,'Linear fit') # 
        fitshape_group3=bayes_fit_class(xData_group3,zData_group3,lambda data,w : wrap_janoshi_local_fun_lin_unpack(data,np.hstack([fitc_group3.w_hat[0:2],w]),yData_group3),Nparr,'Linear fit') # 
        fitshape_group1.baysian_fit(draws,tune,bins,nosubsample=True,x0=[0.001, 0.5, 0.7, 2000])#,nosubsample=True
        fitshape_group2.baysian_fit(draws,tune,bins,nosubsample=True,x0=[0.001, 0.5, 0.7, 2000])#,nosubsample=True
        fitshape_group3.baysian_fit(draws,tune,bins,nosubsample=True,x0=[0.001, 0.5, 0.7, 2000])#,nosubsample=True


