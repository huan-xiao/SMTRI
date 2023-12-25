import numpy as np
from numpy import genfromtxt

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from feature import find_GC_content, count_base_single, count_base_double, count_bp, count_bp_reverse

# retrieve motif sequences from RNA22
case1 = ['(C,G) AGACA (G,C) CCA'] # OC-3, TRAF3
case2 = ['(G,C) UCG (C,G) AAGU'] # 6’-fluorosisomicin, rRNA A-site
case3 = ['(A,U) C (U,A) C', '(G,C) C (G,C) C', '(G,C) A (C,G)'] # Targrpremir-210, miR-210 precursor
case4 = ['(G,C) AACUA (C,G)'] # Isis-11, HCV RNA


columns_names = ['motif','motif_len','motif_GC_content','motif_unpaired_len','motif_unpaired_ratio',\
                 'motif_count_A','motif_count_C','motif_count_G','motif_count_U',\
                 'motif_unpaired_count_A','motif_unpaired_count_C','motif_unpaired_count_G','motif_unpaired_count_U',\
                 'motif_count_AA','motif_count_AC','motif_count_AG','motif_count_AU',\
                 'motif_count_CA','motif_count_CC','motif_count_CG','motif_count_CU',\
                 'motif_count_GA','motif_count_GC','motif_count_GG','motif_count_GU',\
                 'motif_count_UA','motif_count_UC','motif_count_UG','motif_count_UU',\
                 'count_bp_AU','count_bp_UA','count_bp_CG','count_bp_GC','count_bp_GU','count_bp_UG',\
                 'miR_len','miR_ratio','miR_GC_content',\
                 'miR_count_A','miR_count_C','miR_count_G','miR_count_U']


def get_features_for_B(motif_seg): # bulge loop
    # in order, bulge in miRNA
    motif1 = motif_seg[0][1]+motif_seg[1]+motif_seg[2][1]+motif_seg[2][3]+motif_seg[0][3]
    motif_len = len(motif1)
    motif_GC_content = find_GC_content(motif1)/motif_len
    motif_unpaired_len = motif_len-4
    motif_unpaired_ratio = (motif_unpaired_len)/motif_len
    motif_count_A,motif_count_C,motif_count_G,motif_count_U = count_base_single(motif1)
    motif_unpaired_count_A, motif_unpaired_count_C, motif_unpaired_count_G, motif_unpaired_count_U = count_base_single(motif_seg[1])
    motif_count_AA1,motif_count_AC1,motif_count_AG1,motif_count_AU1,\
    motif_count_CA1,motif_count_CC1,motif_count_CG1,motif_count_CU1,\
    motif_count_GA1,motif_count_GC1,motif_count_GG1,motif_count_GU1,\
    motif_count_UA1,motif_count_UC1,motif_count_UG1,motif_count_UU1 = count_base_double(motif1)
    count_bp_AU1,count_bp_UA1,count_bp_CG1,count_bp_GC1,count_bp_GU1,count_bp_UG1 = count_bp(motif_seg[0],motif_seg[2])
    miR_len1 = motif_len-2
    miR_ratio1 = miR_len1/motif_len
    miRNA = motif_seg[0][1]+motif_seg[1]+motif_seg[2][1] # temp
    miR_GC_content1 = find_GC_content(miRNA)/miR_len1
    miR_count_A1, miR_count_C1, miR_count_G1, miR_count_U1 = count_base_single(miRNA)
    
   
    return [motif1,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
            motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
            motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
            motif_count_AA1,motif_count_AC1,motif_count_AG1,motif_count_AU1,\
            motif_count_CA1,motif_count_CC1,motif_count_CG1,motif_count_CU1,\
            motif_count_GA1,motif_count_GC1,motif_count_GG1,motif_count_GU1,\
            motif_count_UA1,motif_count_UC1,motif_count_UG1,motif_count_UU1,\
            count_bp_AU1,count_bp_UA1,count_bp_CG1,count_bp_GC1,count_bp_GU1,count_bp_UG1,\
            miR_len1,miR_ratio1,miR_GC_content1,\
            miR_count_A1,miR_count_C1,miR_count_G1,miR_count_U1]


def get_features_for_I(motif_seg): #  internal loop
    # in order
    motif1= motif_seg[0][1]+motif_seg[1]+motif_seg[2][1]+motif_seg[2][3]+motif_seg[3]+motif_seg[0][3]
    motif_len = len(motif1)
    motif_GC_content = find_GC_content(motif1)/motif_len
    motif_unpaired_len = motif_len-4
    motif_unpaired_ratio = motif_unpaired_len/motif_len
    motif_count_A,motif_count_C,motif_count_G,motif_count_U = count_base_single(motif1)
    motif_unpaired_count_A, motif_unpaired_count_C, motif_unpaired_count_G, motif_unpaired_count_U = count_base_single(motif_seg[1]+motif_seg[3])
    motif_count_AA1,motif_count_AC1,motif_count_AG1,motif_count_AU1,\
    motif_count_CA1,motif_count_CC1,motif_count_CG1,motif_count_CU1,\
    motif_count_GA1,motif_count_GC1,motif_count_GG1,motif_count_GU1,\
    motif_count_UA1,motif_count_UC1,motif_count_UG1,motif_count_UU1 = count_base_double(motif1)
    count_bp_AU1,count_bp_UA1,count_bp_CG1,count_bp_GC1,count_bp_GU1,count_bp_UG1 = count_bp(motif_seg[0],motif_seg[2])
    miR_len1 = len(motif_seg[1])+2
    miR_ratio1 = miR_len1/motif_len
    miRNA = motif_seg[0][1]+motif_seg[1]+motif_seg[2][1] # temp
    miR_GC_content1 = find_GC_content(miRNA)/miR_len1
    miR_count_A1, miR_count_C1, miR_count_G1, miR_count_U1 = count_base_single(miRNA)
      
    return [motif1,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
            motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
            motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
            motif_count_AA1,motif_count_AC1,motif_count_AG1,motif_count_AU1,\
            motif_count_CA1,motif_count_CC1,motif_count_CG1,motif_count_CU1,\
            motif_count_GA1,motif_count_GC1,motif_count_GG1,motif_count_GU1,\
            motif_count_UA1,motif_count_UC1,motif_count_UG1,motif_count_UU1,\
            count_bp_AU1,count_bp_UA1,count_bp_CG1,count_bp_GC1,count_bp_GU1,count_bp_UG1,\
            miR_len1,miR_ratio1,miR_GC_content1,\
            miR_count_A1,miR_count_C1,miR_count_G1,miR_count_U1]


def generate_motif_feature(case): # latent feature + statistical feature

    # generate motif statistical features
    motif_sf = []
    for motif in case:
        motif_seg = motif.split(' ')
        seg_num = len(motif_seg)

        if seg_num==3: # bulge
            motif_sf.append(get_features_for_B(motif_seg))
        elif seg_num==4: # internal loop
            motif_sf.append(get_features_for_I(motif_seg))

    # generate motif latent feature
    dict = {'A': [1,0,0,0,0], 'C': [0,1,0,0,0], 'G': [0,0,1,0,0], 'U': [0,0,0,1,0]}
    AE_model = tf.keras.models.load_model('./model/autoencoder.h5') # if the model cannot be loaded, try downgrade h5py
    
    motifs_onehot = []
    for m in motif_sf:
        m_onehot=[]
        for letter in m[0]:
            m_onehot.append(dict[letter])
        motifs_onehot.append(m_onehot)
    
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(motifs_onehot, maxlen=16,padding="post",value=[0,0,0,0,1])
    motif_ae = AE_model.predict(padded_inputs,verbose=0)

    # combine autoencoder feature with statistical feature
    motif_sf = np.array(motif_sf)[:,1:]
    motif_sf = motif_sf.astype('float64')
    motif_feature = np.concatenate((motif_ae,motif_sf),axis=1)
    
    return motif_feature
    
    
def concatenate_feature(case): # concate motif feature + SM feature
    SM_feature = genfromtxt('./data/SM_feature.csv', delimiter=',', skip_header = 1)
    motif_feature = generate_motif_feature(case)
    
    num_motif = len(motif_feature)
    num_SM = len(SM_feature)
    
    motif_SM = []

    for motif in motif_feature:
        for sm in SM_feature:
            motif_SM.append(np.concatenate((motif, sm)))
    motif_SM = np.array(motif_SM)
    motif_SM = motif_SM.reshape(-1, 1620, 1) # CNN
    
    return motif_SM, num_motif, num_SM
    

if __name__ == "__main__":
    
    
    DNN_model  = tf.keras.models.load_model('./model/best_DNN_model.h5')
    
    num = 0
    for case in [case1, case2, case3, case4]: # case1-5
        num+=1
        print("case "+str(num)+":")
        
        motif_SM, num_motif, num_SM = concatenate_feature(case) 
        results = DNN_model.predict(motif_SM)

        # print results
        total = 0

        for i in range(num_motif):
            for j in range(num_SM):
                index = i*num_SM+j

                if results[index][0]>0.85:
                    print('motif '+str(i)+', SM '+str(j)+': ', end='')
                    print(results[index][0])
                    total = total+1
                    
                    