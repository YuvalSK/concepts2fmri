# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:38:15 2024

add norm before cosine! 

Tutorials on THINGS dataset:
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/fmri_usage.ipynb
- https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/notebooks/working_with_rois.ipynb

@author: YSK
"""
import time
#from os.path import join as pjoin
import os
import shutil

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, correlation, mahalanobis, euclidean
# angle measures = corr is more common in NS, cos in DNN
# magnitude measures = euclidean (non-distributional) is more common, but some use mahalanobis (distributional)
from scipy.stats import permutation_test, spearmanr

#all 209 available ROIs in THINGS dataset
all_rois = ['V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1 (prf)', 'LO2 (prf)',
            'TO1', 'TO2', 'V3b', 'V3a', 'lEBA', 'rEBA', 'lFFA', 'rFFA', 'lOFA', 'rOFA', 
            'lSTS', 'rSTS', 'lPPA', 'rPPA', 'lRSC', 'rRSC', 'lTOS', 'rTOS', 
            'lLOC', 'rLOC', 'IT',
            'glasser-V1', 'glasser-MST', 'glasser-V6', 'glasser-V2', 'glasser-V3', 'glasser-V4', 
            'glasser-V8', 'glasser-4', 'glasser-3b', 'glasser-FEF', 'glasser-PEF', 'glasser-55b', 
            'glasser-V3A', 'glasser-RSC', 'glasser-POS2', 'glasser-V7', 'glasser-IPS1', 
            'glasser-FFC', 'glasser-V3B', 'glasser-LO1', 'glasser-LO2', 'glasser-PIT', 'glasser-MT', 
            'glasser-A1', 'glasser-PSL', 'glasser-SFL', 'glasser-PCV', 'glasser-STV', 'glasser-7Pm', 
            'glasser-7m', 'glasser-POS1', 'glasser-23d', 'glasser-v23ab', 'glasser-d23ab', 'glasser-31pv', 
            'glasser-5m', 'glasser-5mv', 'glasser-23c', 'glasser-5L', 'glasser-24dd', 'glasser-24dv', 
            'glasser-7AL', 'glasser-SCEF', 'glasser-6ma', 'glasser-7Am', 'glasser-7Pl', 'glasser-7PC', 
            'glasser-LIPv', 'glasser-VIP', 'glasser-MIP', 'glasser-1', 'glasser-2', 'glasser-3a', 'glasser-6d', 
            'glasser-6mp', 'glasser-6v', 'glasser-p24pr', 'glasser-33pr', 'glasser-a24pr', 'glasser-p32pr', 
            'glasser-a24', 'glasser-d32', 'glasser-8BM', 'glasser-p32', 'glasser-10r', 'glasser-47m', 
            'glasser-8Av', 'glasser-8Ad', 'glasser-9m', 'glasser-8BL', 'glasser-9p', 'glasser-10d', 
            'glasser-8C', 'glasser-44', 'glasser-45', 'glasser-47l', 'glasser-a47r', 'glasser-6r', 
            'glasser-IFJa', 'glasser-IFJp', 'glasser-IFSp', 'glasser-IFSa','glasser-p9-46v', 
            'glasser-46', 'glasser-a9-46v', 'glasser-9-46d', 'glasser-9a', 'glasser-10v', 
            'glasser-a10p', 'glasser-10pp', 'glasser-11l', 'glasser-13l', 'glasser-OFC', 
            'glasser-47s', 'glasser-LIPd', 'glasser-6a', 'glasser-i6-8', 'glasser-s6-8', 
            'glasser-43', 'glasser-OP4', 'glasser-OP1', 'glasser-OP2-3', 'glasser-52', 
            'glasser-RI', 'glasser-PFcm', 'glasser-PoI2', 'glasser-TA2', 'glasser-FOP4', 
            'glasser-MI', 'glasser-Pir', 'glasser-AVI', 'glasser-AAIC', 'glasser-FOP1', 
            'glasser-FOP3', 'glasser-FOP2', 'glasser-PFt', 'glasser-AIP', 'glasser-EC', 
            'glasser-PreS', 'glasser-H', 'glasser-ProS', 'glasser-PeEc', 'glasser-STGa', 
            'glasser-PBelt', 'glasser-A5', 'glasser-PHA1', 'glasser-PHA3', 'glasser-STSda', 
            'glasser-STSdp', 'glasser-STSvp', 'glasser-TGd', 'glasser-TE1a', 'glasser-TE1p', 
            'glasser-TE2a', 'glasser-TF', 'glasser-TE2p', 'glasser-PHT', 'glasser-PH', 'glasser-TPOJ1', 
            'glasser-TPOJ2', 'glasser-TPOJ3', 'glasser-DVT', 'glasser-PGp', 'glasser-IP2', 'glasser-IP1', 
            'glasser-IP0', 'glasser-PFop', 'glasser-PF', 'glasser-PFm', 'glasser-PGi', 'glasser-PGs', 
            'glasser-V6A', 'glasser-VMV1', 'glasser-VMV3', 'glasser-PHA2', 'glasser-V4t', 'glasser-FST', 
            'glasser-V3CD', 'glasser-LO3', 'glasser-VMV2', 'glasser-31pd', 'glasser-31a', 'glasser-VVC', 
            'glasser-25', 'glasser-s32', 'glasser-pOFC', 'glasser-PoI1', 'glasser-Ig', 'glasser-FOP5', 
            'glasser-p10p', 'glasser-p47r', 'glasser-TGv', 'glasser-MBelt', 'glasser-LBelt', 'glasser-A4', 
            'glasser-STSva', 'glasser-TE1m', 'glasser-PI', 'glasser-a32pr', 'glasser-p24']        

def euc_statistic(x, y):
    return euclidean(x, y)

def correlation_statistic(x, y):
    return correlation(x, y)

def spearmanr_statistic(x, y):
    return spearmanr(x, y)[0]

def main(subjects, rois, alpha, loaded=False, folder="C:/Users/User/Projects/THINGS-data"):
    os.chdir(folder)
    from thingsmri.dataset import ThingsmriLoader
    cs = pd.read_csv("concepts - pilot.csv")
                    
    for sub in subjects:
        dl = ThingsmriLoader(
            thingsmri_dir='.'
        )
            
        if loaded==False: 

            print(f'Loading subject number {sub}...')
            start_time = time.time()
            responses, stimdata, voxdata = dl.load_responses(sub)
            loading_time = (time.time() - start_time)/60
            print(f"-loaded! took {loading_time:.1f} min" )

            '''
            subject 01:
            1) responses (211339 voxels, 9840 trials) 
            2) stimdata (9840 trials, 7 parameters)
            3) voxdata (211339 voxels, 222 parameters)
        
            loading only stim data:
                
                #basedir = os.getcwd()
                #betas_csv_dir = pjoin(basedir, 'betas_csv')
                #stim_f = pjoin(betas_csv_dir, f'sub-{sub}_StimulusMetadata.csv')
                #stimdata = pd.read_csv(stim_f)
                
            #three types of trials: train, test and catch. The latter with non-object
            '''
        
        rois2test = []
        for roi in rois:
            print(f"-{rois.index(roi)+1}/{len(rois)} Analyzing {roi}...")
            if not os.path.isdir(f"figures/subject{sub}/{roi}"):    
                os.mkdir(f"figures/subject{sub}/{roi}")
                
            if roi in list(voxdata.columns.values):
                # selecting all columns in the voxel metadata, and sum over them to create a boolean mask that reflects their union
                roimask = voxdata[[roi]].sum(axis=1).values.astype(bool)
                roidata = responses[roimask] # applying this mask to get the activation of ROIs voxels to each image
                
                #loading pilot data
                concepts = pd.concat([cs['concept 1'], cs['concept 2']])
                #print(f"--fMRI data for ROI={rois} with {roidata.shape[0]} voxels and {roidata.shape[1]} trials \n--pilot data has {len(cs['Concepts'])} chimerical animals out of {len(set(concepts))} unique concepts")
                
                null = []
                df = pd.DataFrame()
                
                for c in set(concepts):
                    #print(f"-'{c}'")
                    query = f'concept == "{c}"'
                    con_indices = stimdata.query(query).index
                    if con_indices.any():
                        #print("---found! plotting...")                        
                        df[c] = roidata[con_indices].mean(axis=1)
                        #plot mean across trials per concept
                        if not os.path.isdir(f"figures/subject{sub}/{roi}"): 
                            plt.clf()
                            plt.bar(x = range(len(df[c])), height= df[c].to_numpy(), color='k')
                            plt.xlabel(f'Voxels [#]');
                            plt.ylabel('Amplitude')
                            plt.title(f"Avg. {roi} activity for '{c}' over {len(con_indices)} trials")
                            plt.savefig(f'figures/subject{sub}/{roi}/{c}', dpi=500)
                    else:
                        null.append(c)
                                
                    #print(len(set(null)))
                    #13-{'ostrich', 'elephant ', 'duck', 'chicken', 'peacock', 'checken', 'seal', 'mouse', 'gorilla', 'eagle', 'rooster', 'owl', 'swan', 'fox'}
                    
                df.to_csv(f"figures/subject{sub}/{roi}/voxels.csv")
                
                #part B - running similarity analysis
                # Custom function for cosine distance without the axis argument
                ex_concepts = df.columns.values[1:]
                
                results = []
                null = []        
                for i, c in enumerate(cs['Concepts']):
                    #print(f"-unpacking {c}")
                    c1, c2 = cs.iloc[i, 2], cs.iloc[i, 3]
                    if c1 in ex_concepts and c2 in ex_concepts:
                        #print(f"--analysing...")
                        res_euc = permutation_test(data=(df[c1], df[c2]), 
                                                   statistic=euc_statistic, 
                                                   vectorized=False, 
                                                   n_resamples=5000, 
                                                   alternative='two-sided')
                        
                        res_corr = permutation_test(data=(df[c1], df[c2]), 
                                                    statistic = correlation_statistic, 
                                                    vectorized=False,
                                                    n_resamples=5000, 
                                                    alternative='two-sided')
                        
                        results.append([c, res_euc.statistic, res_corr.statistic])
                    else:
                        #print("--not found!")
                        null.append(c)
                
                #print(f"-{len(results)} chimerical animals have fMRI data as concepts!")
                ## sub1 = 46, sub2 = 46, sub3 = 46
                
                rs = pd.DataFrame(results, columns=['Concept', 'Euclidean', 'Pearson'])
                rs.to_csv(f"figures/subject{sub}/{roi}/similarity - pilot.csv")
                
                ### running stats without loading all data ###
                #sub = '03'
                #rois = ['rLOC', 'lLOC']
                #cs = pd.read_csv("concepts - pilot.csv")
                #df = pd.read_csv(f"figures/subject{sub}_{rois}_voxels.csv")
                #rs = pd.read_csv(f"figures/subject{sub}_{rois}_similarity - pilot.csv")
                
                cs_filtered = cs.loc[cs['Concepts'].isin(rs["Concept"])]                   
                corr_tests = ['clip-real','clip-vis-sim'] # May be more robust to test all concepts vs. CLIP vis-sim                
                results_sum = []
                for t in corr_tests:
                    res_euc = permutation_test(data=(cs_filtered[t], rs["Euclidean"]), 
                                                statistic = spearmanr_statistic, 
                                                vectorized=False,
                                                n_resamples=5000, 
                                                alternative='two-sided')
                    #print(f"{t} and cosine:\n --r = {res_cos.statistic:.3f}, p = {res_cos.pvalue:.5f}") 
                    if res_euc.pvalue < alpha:
                        #print(f"---{t} and cosine:\n --r = {res_cos.statistic:.3f}, p = {res_cos.pvalue:.5f}") 
                        results_sum.append(f"s{sub}, {roi}, euc, {t}, {res_euc.statistic}, {res_euc.pvalue}") 

                        plt.clf()
                        plt.hist(res_euc.null_distribution, bins=50)
                        plt.axvline(x=res_euc.statistic, color='r')
                        plt.title(f"{t} and euc:\nr = {res_euc.statistic:.3f}, p = {res_euc.pvalue:.5f}")
                        plt.xlabel("Correlation [rho]")
                        plt.ylabel("Frequency")
                        plt.xlim([-1, 1])
                        plt.savefig(f"figures/subject{sub}/{roi}/{t} and euc.png", dpi=800)
                        
                for t in corr_tests:
                    res_corr = permutation_test(data=(cs_filtered[t], rs["Pearson"]), 
                                            statistic = spearmanr_statistic, 
                                            vectorized=False,
                                            n_resamples=5000, 
                                            alternative='two-sided')
                    if res_corr.pvalue < alpha:
                        #print(f"---{t} and corr:\n --r = {res_corr.statistic:.3f}, p = {res_corr.pvalue:.5f}") 
                        results_sum.append(f"s{sub}, {roi}, pearson, {t}, {res_corr.statistic}, {res_corr.pvalue}")
                        plt.clf()
                        plt.hist(res_corr.null_distribution, bins=50)
                        plt.axvline(x=res_corr.statistic, color='r')
                        plt.title(f"{t} and correlation:\nr = {res_corr.statistic:.3f}, p = {res_corr.pvalue:.5f}")
                        plt.xlabel("Correlation [rho]")
                        plt.ylabel("Frequency")
                        plt.xlim([-1, 1])
                        plt.savefig(f"figures/subject{sub}/{roi}/{t} and corr.png",dpi=800)
                
                if results_sum:
                    print(results_sum)
                    rois2test.append(roi)
                    #only if found something (fs) in that ROI 
                    fs = pd.DataFrame(results_sum)
                    fs.to_csv(f"figures/subject{sub}/{roi}/results.csv")
                else:
                    shutil.rmtree(f"figures/subject{sub}/{roi}")
        
        if len(subjects)==1:
            return rois2test      

#to minimize comparisons, we do cross validation:
## part A - exploring a single subject
### rois = ['rLOC', 'lLOC'] #hypothesis-based, inconsistent results

s_test = ['03', '02', '01']
rois2test = main(subjects = [s_test[0]],
                rois = all_rois,
                alpha = 0.01
                loaded=False
                )

## part B - cross validation on the other subjects
folder = f"C:/Users/User/Projects/THINGS-data/figures/subject{s_test[0]}"
rois2test = [ item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item)) ]

#cross validation on the other subjects
main(subjects = s_test[1:],
     rois = rois2test, 
     alpha = 0.01,
     loaded=False
     )









