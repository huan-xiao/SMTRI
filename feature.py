# to generate and concatenate all features (2 features for RNA, 2 features for SM).
# the feature order in csv: RNA latent features, RNA statistical features, SM fingerprints, SM descriptors.

import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from e3fp.pipeline import fprints_from_smiles


# motif features
columns_names = ['index', 'motif','motif_len','motif_GC_content','motif_unpaired_len','motif_unpaired_ratio',\
                 'motif_count_A','motif_count_C','motif_count_G','motif_count_U',\
                 'motif_unpaired_count_A','motif_unpaired_count_C','motif_unpaired_count_G','motif_unpaired_count_U',\
                 'motif_count_AA','motif_count_AC','motif_count_AG','motif_count_AU',\
                 'motif_count_CA','motif_count_CC','motif_count_CG','motif_count_CU',\
                 'motif_count_GA','motif_count_GC','motif_count_GG','motif_count_GU',\
                 'motif_count_UA','motif_count_UC','motif_count_UG','motif_count_UU',\
                 'count_bp_AU','count_bp_UA','count_bp_CG','count_bp_GC','count_bp_GU','count_bp_UG',\
                 'miR_len','miR_ratio','miR_GC_content',\
                 'miR_count_A','miR_count_C','miR_count_G','miR_count_U']



def get_SM_fingerprints(data_df):
    confgen_params = {'num_conf': 10, 'first': 1}
    fprint_params = {'bits': 1024, 'rdkit_invariants': True, 'first': 1}

    fp = []
    num=0
    for index, row in data_df.iterrows():
        Flag=True

        try:
            fprints = fprints_from_smiles(row['SMILES'], num, confgen_params=confgen_params, fprint_params=fprint_params)
        except:
            try:
                confgen_params_2 = {'num_conf': 30, 'first': 1}
                fprints = fprints_from_smiles(row['SMILES'], num, confgen_params=confgen_params_2, fprint_params=fprint_params)
            except:
                Flag=False
                data_df.drop(index, axis=0, inplace=True)
                print('cannot genrate fingerprints:', end='\t')
                print(num)

        # if fingerprints can be generated
        if Flag:
            fprints_bit = fprints[0].to_vector(sparse=False)*1
            fp.append(fprints_bit)

        # next
        num+=1
        
        fp_df = pd.DataFrame(fp)
        column_names = ['fp_' + str(column_name) for column_name in fp_df.columns.values]
        fp_df.columns = column_names
        
    return fp_df, data_df # some molecule cannot generate fp
        
        
        
def get_SM_descriptors(data_df):
    descriptors = []
    
    for index, row in data_df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        properties = rdMolDescriptors.Properties()
        mol_desc = list(properties.ComputeProperties(mol))
        descriptors.append(mol_desc)
        
    des_df = pd.DataFrame(descriptors, columns=['exactmw',  'amw',  'lipinskiHBA',  'lipinskiHBD',  'NumRotatableBonds',  'NumHBD',  'NumHBA',  'NumHeavyAtoms',  'NumAtoms',  'NumHeteroatoms',  'NumAmideBonds',  'FractionCSP3',  'NumRings',  'NumAromaticRings',  'NumAliphaticRings',  'NumSaturatedRings',  'NumHeterocycles',  'NumAromaticHeterocycles',  'NumSaturatedHeterocycles',  'NumAliphaticHeterocycles',  'NumSpiroAtoms',  'NumBridgeheadAtoms',  'NumAtomStereoCenters',  'NumUnspecifiedAtomStereoCenters',  'labuteASA',  'tpsa',  'CrippenClogP',  'CrippenMR',  'chi0v',  'chi1v',  'chi2v',  'chi3v',  'chi4v',  'chi0n',  'chi1n',  'chi2n',  'chi3n',  'chi4n',  'hallKierAlpha',  'kappa1',  'kappa2',  'kappa3',  'Phi'])
    
    return des_df
   

    
def find_GC_content(motif):
    num = 0
    for base in motif:
        if base == 'C' or base == 'G':
            num += 1
    return num



def count_base_single(motif):
    count_A = 0
    count_C = 0
    count_G = 0
    count_U = 0
    
    for base in motif:
        if base == 'A':
            count_A += 1
        elif base == 'C':
            count_C += 1
        elif base == 'G':
            count_G += 1
        elif base == 'U':
            count_U += 1
    
    return count_A,count_C,count_G,count_U
         
    

def count_base_double(motif):
    count_AA = 0
    count_AC = 0
    count_AG = 0
    count_AU = 0
    count_CA = 0
    count_CC = 0
    count_CG = 0
    count_CU = 0
    count_GA = 0
    count_GC = 0
    count_GG = 0
    count_GU = 0
    count_UA = 0
    count_UC = 0
    count_UG = 0
    count_UU = 0
    
    for win in range(len(motif)-1):
        if motif[win:win+2] == 'AA':
            count_AA += 1
        elif motif[win:win+2] == 'AC':
            count_AC += 1
        elif motif[win:win+2] == 'AG':
            count_AG += 1
        elif motif[win:win+2] == 'AU':
            count_AU += 1
        elif motif[win:win+2] == 'CA':
            count_CA += 1
        elif motif[win:win+2] == 'CC':
            count_CC += 1
        elif motif[win:win+2] == 'CG':
            count_CG += 1
        elif motif[win:win+2] == 'CU':
            count_CU += 1
        elif motif[win:win+2] == 'GA':
            count_GA += 1
        elif motif[win:win+2] == 'GC':
            count_GC += 1
        elif motif[win:win+2] == 'GG':
            count_GG += 1
        elif motif[win:win+2] == 'GU':
            count_GU += 1
        elif motif[win:win+2] == 'UA':
            count_UA += 1
        elif motif[win:win+2] == 'UC':
            count_UC += 1
        elif motif[win:win+2] == 'UG':
            count_UG += 1
        elif motif[win:win+2] == 'UU':
            count_UU += 1
    
    return count_AA,count_AC,count_AG,count_AU,\
           count_CA,count_CC,count_CG,count_CU,\
           count_GA,count_GC,count_GG,count_GU,\
           count_UA,count_UC,count_UG,count_UU


            
def count_bp(base_pair_seg1, base_pair_seg2=None):
    count_bp_AU = 0
    count_bp_UA = 0
    count_bp_CG = 0
    count_bp_GC = 0
    count_bp_GU = 0
    count_bp_UG = 0
    
    base_pair1 = base_pair_seg1[1]+base_pair_seg1[3]
    
    if base_pair1 == 'AU':
        count_bp_AU += 1
    elif base_pair1 == 'UA':
        count_bp_UA += 1   
    elif base_pair1 == 'CG':
        count_bp_CG += 1   
    elif base_pair1 == 'GC':
        count_bp_GC += 1   
    elif base_pair1 == 'GU':
        count_bp_GU += 1   
    elif base_pair1 == 'UG':
        count_bp_UG += 1   

    if base_pair_seg2 == None: #hairpin
        return count_bp_AU,count_bp_UA,count_bp_CG,count_bp_GC,count_bp_GU,count_bp_UG
    
    # else
    base_pair2 = base_pair_seg2[1]+base_pair_seg2[3]
    
    if base_pair2 == 'AU':
        count_bp_AU += 1
    elif base_pair2 == 'UA':
        count_bp_UA += 1   
    elif base_pair2 == 'CG':
        count_bp_CG += 1   
    elif base_pair2 == 'GC':
        count_bp_GC += 1   
    elif base_pair2 == 'GU':
        count_bp_GU += 1   
    elif base_pair2 == 'UG':
        count_bp_UG += 1   
    
    return count_bp_AU,count_bp_UA,count_bp_CG,count_bp_GC,count_bp_GU,count_bp_UG



def count_bp_reverse(count_bp_AU1, count_bp_UA1, count_bp_CG1, count_bp_GC1, count_bp_GU1, count_bp_UG1):
    count_bp_AU2 = 0
    count_bp_UA2 = 0
    count_bp_CG2 = 0
    count_bp_GC2 = 0
    count_bp_GU2 = 0
    count_bp_UG2 = 0
    
    if count_bp_AU1 > 0:
        count_bp_UA2 = count_bp_AU1

    if count_bp_UA1 > 0:
        count_bp_AU2 = count_bp_UA1

    if count_bp_CG1 > 0:
        count_bp_GC2 = count_bp_CG1

    if count_bp_GC1 > 0:
        count_bp_CG2 = count_bp_GC1

    if count_bp_GU1 > 0:
        count_bp_UG2 = count_bp_GU1

    if count_bp_UG1 > 0:
        count_bp_GU2 = count_bp_UG1

    return count_bp_AU2,count_bp_UA2,count_bp_CG2,count_bp_GC2,count_bp_GU2,count_bp_UG2
    
    
    
def get_features_for_E(index, motif_seg): # external loop
    motif = motif_seg[0]
    motif_len = len(motif)
    motif_GC_content = find_GC_content(motif)/motif_len
    motif_unpaired_len = motif_len
    motif_unpaired_ratio = 1
    motif_count_A,motif_count_C,motif_count_G,motif_count_U = count_base_single(motif)
    motif_unpaired_count_A = motif_count_A
    motif_unpaired_count_C = motif_count_C
    motif_unpaired_count_G = motif_count_G
    motif_unpaired_count_U = motif_count_U
    motif_count_AA,motif_count_AC,motif_count_AG,motif_count_AU,\
    motif_count_CA,motif_count_CC,motif_count_CG,motif_count_CU,\
    motif_count_GA,motif_count_GC,motif_count_GG,motif_count_GU,\
    motif_count_UA,motif_count_UC,motif_count_UG,motif_count_UU = count_base_double(motif)
    count_bp_AU = 0
    count_bp_UA = 0
    count_bp_CG = 0
    count_bp_GC = 0
    count_bp_GU = 0
    count_bp_UG = 0
    miR_len = motif_len/2 # miRNA-mRNA do have this motif, mostly at the end of the sequence
    miR_ratio = 0.5
    miR_GC_content = find_GC_content(motif[miR_len:])/miR_len # use the second half, because in miR-mR we count external loop as 5'-mR-miR-3'
    miR_count_A, miR_count_C, miR_count_G, miR_count_U = count_base_single(motif[miR_len:])
    
    return pd.DataFrame([[index, motif,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
                          motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
                          motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
                          motif_count_AA,motif_count_AC,motif_count_AG,motif_count_AU,\
                          motif_count_CA,motif_count_CC,motif_count_CG,motif_count_CU,\
                          motif_count_GA,motif_count_GC,motif_count_GG,motif_count_GU,\
                          motif_count_UA,motif_count_UC,motif_count_UG,motif_count_UU,\
                          count_bp_AU,count_bp_UA,count_bp_CG,count_bp_GC,count_bp_GU,count_bp_UG,\
                          miR_len,miR_ratio,miR_GC_content,\
                          miR_count_A,miR_count_C,miR_count_G,miR_count_U]],columns=columns_names)



def get_features_for_H(index, motif_seg): # hairpin loop, testing set won't have hairpin, so the feature is desinged for training's convenience
    motif = motif_seg[0][1]+motif_seg[1]+motif_seg[0][3]
    motif_len = len(motif)
    motif_GC_content = find_GC_content(motif)/motif_len
    motif_unpaired_len = motif_len-2
    motif_unpaired_ratio = motif_unpaired_len/motif_len
    motif_count_A,motif_count_C,motif_count_G,motif_count_U = count_base_single(motif)
    motif_unpaired_count_A, motif_unpaired_count_C, motif_unpaired_count_G, motif_unpaired_count_U = count_base_single(motif_seg[1])
    motif_count_AA,motif_count_AC,motif_count_AG,motif_count_AU,\
    motif_count_CA,motif_count_CC,motif_count_CG,motif_count_CU,\
    motif_count_GA,motif_count_GC,motif_count_GG,motif_count_GU,\
    motif_count_UA,motif_count_UC,motif_count_UG,motif_count_UU = count_base_double(motif)
    count_bp_AU,count_bp_UA,count_bp_CG,count_bp_GC,count_bp_GU,count_bp_UG = count_bp(motif_seg[0])
    miR_len = motif_len # hairpin will be considered as from miRNA, it is not necessary to create another motif feature
    miR_ratio = 1
    miR_GC_content = motif_GC_content
    miR_count_A = motif_count_A
    miR_count_C = motif_count_C
    miR_count_G = motif_count_G
    miR_count_U = motif_count_U
    
    return pd.DataFrame([[index, motif,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
                          motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
                          motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
                          motif_count_AA,motif_count_AC,motif_count_AG,motif_count_AU,\
                          motif_count_CA,motif_count_CC,motif_count_CG,motif_count_CU,\
                          motif_count_GA,motif_count_GC,motif_count_GG,motif_count_GU,\
                          motif_count_UA,motif_count_UC,motif_count_UG,motif_count_UU,\
                          count_bp_AU,count_bp_UA,count_bp_CG,count_bp_GC,count_bp_GU,count_bp_UG,\
                          miR_len,miR_ratio,miR_GC_content,\
                          miR_count_A,miR_count_C,miR_count_G,miR_count_U]],columns=columns_names)



def get_features_for_B(index, motif_seg): # bulge loop
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
    
    # in reverse order, bulge in mRNA
    motif2 = motif_seg[2][3]+motif_seg[0][3]+motif_seg[0][1]+motif_seg[1]+motif_seg[2][1]
    motif_count_AA2,motif_count_AC2,motif_count_AG2,motif_count_AU2,\
    motif_count_CA2,motif_count_CC2,motif_count_CG2,motif_count_CU2,\
    motif_count_GA2,motif_count_GC2,motif_count_GG2,motif_count_GU2,\
    motif_count_UA2,motif_count_UC2,motif_count_UG2,motif_count_UU2 = count_base_double(motif2)
    count_bp_AU2,count_bp_UA2,count_bp_CG2,count_bp_GC2,count_bp_GU2,count_bp_UG2 = count_bp_reverse(\
                                                                                        count_bp_AU1,count_bp_UA1,\
                                                                                        count_bp_CG1,count_bp_GC1,\
                                                                                        count_bp_GU1,count_bp_UG1)
    miR_len2 = 2
    miR_ratio2 = miR_len2/motif_len
    miRNA = motif_seg[2][3]+motif_seg[0][3] # temp
    miR_GC_content2 = find_GC_content(miRNA)/miR_len2
    miR_count_A2, miR_count_C2, miR_count_G2, miR_count_U2 = count_base_single(miRNA)
    
    return pd.DataFrame([[index, motif1,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
                          motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
                          motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
                          motif_count_AA1,motif_count_AC1,motif_count_AG1,motif_count_AU1,\
                          motif_count_CA1,motif_count_CC1,motif_count_CG1,motif_count_CU1,\
                          motif_count_GA1,motif_count_GC1,motif_count_GG1,motif_count_GU1,\
                          motif_count_UA1,motif_count_UC1,motif_count_UG1,motif_count_UU1,\
                          count_bp_AU1,count_bp_UA1,count_bp_CG1,count_bp_GC1,count_bp_GU1,count_bp_UG1,\
                          miR_len1,miR_ratio1,miR_GC_content1,\
                          miR_count_A1,miR_count_C1,miR_count_G1,miR_count_U1]],columns=columns_names),\
           pd.DataFrame([[index, motif2,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
                          motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
                          motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
                          motif_count_AA2,motif_count_AC2,motif_count_AG2,motif_count_AU2,\
                          motif_count_CA2,motif_count_CC2,motif_count_CG2,motif_count_CU2,\
                          motif_count_GA2,motif_count_GC2,motif_count_GG2,motif_count_GU2,\
                          motif_count_UA2,motif_count_UC2,motif_count_UG2,motif_count_UU2,\
                          count_bp_AU2,count_bp_UA2,count_bp_CG2,count_bp_GC2,count_bp_GU2,count_bp_UG2,\
                          miR_len2,miR_ratio2,miR_GC_content2,\
                          miR_count_A2,miR_count_C2,miR_count_G2,miR_count_U2]],columns=columns_names)
    
    
    
def get_features_for_I(index, motif_seg): #  internal loop
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

    # in reverse order
    motif2= motif_seg[2][3]+motif_seg[3]+motif_seg[0][3]+motif_seg[0][1]+motif_seg[1]+motif_seg[2][1]
    motif_count_AA2,motif_count_AC2,motif_count_AG2,motif_count_AU2,\
    motif_count_CA2,motif_count_CC2,motif_count_CG2,motif_count_CU2,\
    motif_count_GA2,motif_count_GC2,motif_count_GG2,motif_count_GU2,\
    motif_count_UA2,motif_count_UC2,motif_count_UG2,motif_count_UU2 = count_base_double(motif2)
    count_bp_AU2,count_bp_UA2,count_bp_CG2,count_bp_GC2,count_bp_GU2,count_bp_UG2 = count_bp_reverse(\
                                                                                        count_bp_AU1,count_bp_UA1,\
                                                                                        count_bp_CG1,count_bp_GC1,\
                                                                                        count_bp_GU1,count_bp_UG1)
    miR_len2 = len(motif_seg[3])+2
    miR_ratio2 = miR_len2/motif_len
    miRNA = motif_seg[2][3]+motif_seg[3]+motif_seg[0][3] # temp
    miR_GC_content2 = find_GC_content(miRNA)/miR_len2
    miR_count_A2, miR_count_C2, miR_count_G2, miR_count_U2 = count_base_single(miRNA)

      
    return pd.DataFrame([[index, motif1,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
                          motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
                          motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
                          motif_count_AA1,motif_count_AC1,motif_count_AG1,motif_count_AU1,\
                          motif_count_CA1,motif_count_CC1,motif_count_CG1,motif_count_CU1,\
                          motif_count_GA1,motif_count_GC1,motif_count_GG1,motif_count_GU1,\
                          motif_count_UA1,motif_count_UC1,motif_count_UG1,motif_count_UU1,\
                          count_bp_AU1,count_bp_UA1,count_bp_CG1,count_bp_GC1,count_bp_GU1,count_bp_UG1,\
                          miR_len1,miR_ratio1,miR_GC_content1,\
                          miR_count_A1,miR_count_C1,miR_count_G1,miR_count_U1]],columns=columns_names),\
           pd.DataFrame([[index, motif2,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
                          motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
                          motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
                          motif_count_AA2,motif_count_AC2,motif_count_AG2,motif_count_AU2,\
                          motif_count_CA2,motif_count_CC2,motif_count_CG2,motif_count_CU2,\
                          motif_count_GA2,motif_count_GC2,motif_count_GG2,motif_count_GU2,\
                          motif_count_UA2,motif_count_UC2,motif_count_UG2,motif_count_UU2,\
                          count_bp_AU2,count_bp_UA2,count_bp_CG2,count_bp_GC2,count_bp_GU2,count_bp_UG2,\
                          miR_len2,miR_ratio2,miR_GC_content2,\
                          miR_count_A2,miR_count_C2,miR_count_G2,miR_count_U2]],columns=columns_names)
    

    
def get_features_for_M(index, motif_seg): # multibranch loop, I can generate several, but since the data is not reliable, for penalty, I just generate one
    # form the linear motif
    seg_num = len(motif_seg)
    motif = motif_seg[0][1]
    motif_mid = ''
    
    for i in range(1, seg_num):
        if motif_seg[i].find('(')!=-1:
            motif_mid += motif_seg[i][1]+motif_seg[i][3]
        else:
            motif_mid += motif_seg[i]
    motif += motif_mid + motif_seg[0][3]
        
    motif_len = len(motif)
    motif_GC_content = find_GC_content(motif)/motif_len 
    motif_unpaired_len = motif_len-2
    motif_unpaired_ratio = motif_unpaired_len/motif_len # treat it as hairpin
    motif_count_A,motif_count_C,motif_count_G,motif_count_U = count_base_single(motif)
    motif_unpaired_count_A, motif_unpaired_count_C, motif_unpaired_count_G, motif_unpaired_count_U = count_base_single(motif_mid)
    motif_count_AA,motif_count_AC,motif_count_AG,motif_count_AU,\
    motif_count_CA,motif_count_CC,motif_count_CG,motif_count_CU,\
    motif_count_GA,motif_count_GC,motif_count_GG,motif_count_GU,\
    motif_count_UA,motif_count_UC,motif_count_UG,motif_count_UU = count_base_double(motif)
    count_bp_AU,count_bp_UA,count_bp_CG,count_bp_GC,count_bp_GU,count_bp_UG = count_bp(motif_seg[0])
    miR_len = motif_len-2 # treat it as whole miRNA, like hairpin
    miR_ratio = miR_len/motif_len
    miRNA = motif_mid # temp
    miR_GC_content = find_GC_content(miRNA)/miR_len
    miR_count_A, miR_count_C, miR_count_G, miR_count_U= count_base_single(miRNA)
     
    return pd.DataFrame([[index, motif,motif_len,motif_GC_content,motif_unpaired_len,motif_unpaired_ratio,\
                          motif_count_A,motif_count_C,motif_count_G,motif_count_U,\
                          motif_unpaired_count_A,motif_unpaired_count_C,motif_unpaired_count_G,motif_unpaired_count_U,\
                          motif_count_AA,motif_count_AC,motif_count_AG,motif_count_AU,\
                          motif_count_CA,motif_count_CC,motif_count_CG,motif_count_CU,\
                          motif_count_GA,motif_count_GC,motif_count_GG,motif_count_GU,\
                          motif_count_UA,motif_count_UC,motif_count_UG,motif_count_UU,\
                          count_bp_AU,count_bp_UA,count_bp_CG,count_bp_GC,count_bp_GU,count_bp_UG,\
                          miR_len,miR_ratio,miR_GC_content,\
                          miR_count_A,miR_count_C,miR_count_G,miR_count_U]],columns=columns_names)



def get_RNA_statistical_feature(data_df):
    df = pd.DataFrame(columns=columns_names)  # output
    
    for index, row in data_df.iterrows():
        print(index, end='\t', flush=True)# otherwise it won't printing
        motif = row['motif']
        motif_seg = motif.split(' ')
        seg_num = len(motif_seg)

        if seg_num==1: # external loop
            df = pd.concat([df, get_features_for_E(index, motif_seg)],ignore_index=True)
        elif seg_num==2: # hairpin
            df = pd.concat([df, get_features_for_H(index, motif_seg)],ignore_index=True)
        elif seg_num==3: # bulge
            df1, df2 = get_features_for_B(index, motif_seg)
            df = pd.concat([df, df1],ignore_index=True)
            df = pd.concat([df, df2],ignore_index=True)
        elif seg_num==4: # internal loop
            df1, df2 = get_features_for_I(index, motif_seg)
            df = pd.concat([df, df1],ignore_index=True)
            df = pd.concat([df, df2],ignore_index=True)
        else: # multibranch loop
            df = pd.concat([df, get_features_for_M(index, motif_seg)],ignore_index=True)
    
    return df
    
    
    
def get_RNA_latent_feature(data_motif): # input: sequence, via: autoencoder, output: 512-bits double float
    dict = {'A': [1,0,0,0,0], 'C': [0,1,0,0,0], 'G': [0,0,1,0,0], 'U': [0,0,0,1,0]}
    
    motifs = []

    for line in data_motif:
        motif=[]
        for letter in line:
            motif.append(dict[letter])
        motifs.append(motif)
    
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(motifs, maxlen=16,padding="post",value=[0,0,0,0,1])
    
    model_encoder = tf.keras.models.load_model('./autoencoder.h5')
    y = model_encoder.predict(padded_inputs)
    
    df_y = pd.DataFrame(y)
    column_names = ['ae_' + str(column_name) for column_name in df_y.columns.values]
    df_y.columns = column_names
    
    return df_y
    
    
    
if __name__ == "__main__": # the four steps can be done separately
    #df_original = pd.read_csv('./data/RNA_motif-SM_SMILES.csv')
    
    print('################## Start getting SM fingerprint features... ##################') # 1024-bits
    #df_fp, df_selected = get_SM_fingerprints(df_original)
    #df_fp.to_csv('./data/saved_fp.csv', index=False)
    #df_selected.to_csv('./data/save_selected.csv', index=False)
    
    #df_fp = pd.read_csv('./data/saved_fp.csv')
    #df_selected = pd.read_csv('./data/saved_selected.csv')
    
    print('################## Start getting SM descriptor features... ##################') # 43-bits
    #df_dp = get_SM_descriptors(df_selected)
    #df_fp_dp = pd.concat([df_fp, df_dp],axis=1)
    #df_fp_dp.to_csv('./data/saved_fp_dp.csv', index=False)
    
    #df_fp_dp = pd.read_csv('./data/saved_fp_dp.csv')
    
    print('################## Start getting RNA statistical features... ##################') # 41-bits, but add one colume of motif sequences
    #df_st = get_RNA_statistical_feature(df_selected)
    #df_st.to_csv('./data/saved_st.csv', index=False)
    
    #df_st = pd.read_csv('./data/saved_st.csv')
    
    print('################## Start getting RNA latent features... ##################') # 512-bits
    #df_lt = get_RNA_latent_feature(df_st['motif'])
    #df_st = df_st.drop('motif', axis=1) # drop original motif column
    #df_lt_st = pd.concat([df_lt, df_st],axis=1)
    #df_lt_st.to_csv('./data/saved_lt_st.csv', index=False)
    
    # save to file
    print('################## Finalizing file... ##################') # 1620-bits
    df_lt_st = pd.read_csv('./data/saved_lt_st.csv')
    df_fp_dp = pd.read_csv('./data/saved_fp_dp.csv')
    df_selected = pd.read_csv('./data/saved_selected.csv')
    
    df_fp_dp = pd.concat([df_fp_dp, df_selected['target']],axis=1)
    
    # adjust data
    df_fp_dp_renew = pd.DataFrame(columns = df_fp_dp.columns)
    for index in df_lt_st['index']:
        df_fp_dp_renew = pd.concat([df_fp_dp_renew, df_fp_dp.iloc[[index]]],ignore_index=True)
    
    df_lt_st = df_lt_st.drop('index', axis=1) # drop original index column
    df_lt_st_fp_dp = pd.concat([df_lt_st, df_fp_dp_renew],axis=1)
    
    # save to training set
    df_lt_st_fp_dp.to_csv('./data/training_set.csv', index=False)
    