from utils.process_data import *
from utils.smile_rel_dist_interpreter import *
from base_line_models import *
from drug_transformer import *
import scipy.stats
from sklearn.mixture import BayesianGaussianMixture
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from collections import Counter
#import keras_nlp
from tensorflow.keras import initializers
import json
tf.keras.utils.set_random_seed(812)
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
from random import seed
from random import sample

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
#import shap
from sklearn.tree import DecisionTreeRegressor
import selfies as sf
import numpy as np
import Geneformer as ge
import gseapy as gp
from category_encoders import BinaryEncoder
#from geneformer.pretrainer import token_dictionary

ensemble_id = pyreadr.read_r('/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/LCCL_input/RNA-CCLE_RNAseq.annot.rds')[None]
gene_expression = gene_expression.set_index("CCLE_ID")

evidence_drug = ['Erlotinib','Irinotecan','lapatinib','nutlin-3','NVP-TAE684','PD-0332991','PLX-4720','sorafenib','topotecan']
evidence_pathway = ['R-HSA-182971 EGFR downregulation', 'HALLMARK_G2M_CHECKPOINT','R-HSA-182971 EGFR downregulation',
                   'R-HSA-5633007 Regulation of TP53 Activity','R-HSA-201556 Signaling by ALK','HALLMARK_E2F_TARGETS',
                   'R-HSA-5675221 Negative regulation of MAPK pathway','WP4685_20240211 Melanoma', 'HALLMARK_G2M_CHECKPOINT']
df_evidence_pathway = pd.DataFrame(list(zip(list(evidence_drug), list(evidence_pathway))),columns=['drug', 'target_pathway'])

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


prior_knowledge_drug_gene = pd.read_table('canonical_smiles.Tingyi_gene.pccompound.gene_id.interaction_score.txt',sep="\t",on_bad_lines='skip',header=None)

P = np.zeros((1, 100, 60))
XX = np.arange(100, dtype=np.float32).reshape(
    -1,1)/np.power(1000, np.arange(
        0, 60, 2, dtype=np.float32) / 60)
P[:, :, 0::2] = np.sin(XX)
P[:, :, 1::2] = np.cos(XX)
#P[0][0] = np.zeros((60))
#shape_X = tf.shape(X)
#X = tf.math.l2_normalize(X, axis=-1)
P = tf.cast(tf.math.l2_normalize(P[:, :100,:], axis=-1), 
    dtype=tf.float32)
edge_type_dict = np.zeros((5,5))
gene_expression_bin_dict = np.zeros((4,4))
gene_mutation_dict = np.zeros((2,2))
for i in range(5):
    edge_type_dict[i,i] = 1
edge_type_dict = tf.cast(edge_type_dict,dtype=tf.float32)
for i in range(4):
    gene_expression_bin_dict[i,i] = 1
gene_expression_bin_dict = tf.cast(gene_expression_bin_dict,dtype=tf.float32)
for i in range(2):
    gene_mutation_dict[i,i] = 1
gene_mutation_dict = tf.cast(gene_mutation_dict, dtype=tf.float32)

def filtering_raw_gene_expression(gene_expression: pd.DataFrame)->pd.DataFrame:
    """
    Compute the variance of each gene expression, and also return 
    the zero amount of gene expression
    
    Parameters:
    -----------
    gene_expression: cell line gene expression input
    
    Returns:
    --------
    dataframe with gene expression variance and zero amount
    """
    std_list = []
    zeros_list = []
    filtered_list = []
    #filtered_index_list = [] 
    std_threshold = 1
    zero_threshold = 250
    gene_names = gene_expression.columns
    index = 0
    for i in gene_names:
        #print(index)
        std = np.nanstd(gene_expression[i])
        std_list.append(std)
        zeros_num = list(gene_expression[i]).count(0)
        zeros_list.append(zeros_num)
        if std < std_threshold or zeros_num > zero_threshold:
            #gene_expression = gene_expression.drop([i],axis=1)
            filtered_list.append(i)
            #filtered_index_list.append(index)
            #print("im here in condition")
        #print(index)
    index+= 1
    gene_expression = gene_expression.drop(filtered_list,axis=1)
    
    return gene_expression

gene_filtered_var = filtering_raw_gene_expression(gene_expression)

drug_ic50_value_whole = []
drug_name_whole = []
gene_expression_value_avail = []
gene_expression_value_whole = []
cell_line_name_avail = []
cell_line_name_whole = []
cell_line_name = cell_line_drug["Cell_line_Name"]
for i in list(cell_line_name):
    try:
        gene_expression_value_avail.append(gene_expression.loc[i])
        cell_line_name_avail.append(i)
    except:
        continue

mutation = pyreadr.read_r('/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/lung_and_all_processed_data/CCLE/driver_mutations_all.rds')[None]
mutation.set_index("CCLE_ID", inplace =True)
mutation_avail_filter = mutation.loc[cell_line_name_avail].replace('',0)
mutation_avail_filter[mutation_avail_filter != 0] = 1

drug_prior = pd.read_csv('drug.pccompound.TARGET.TARGET_PATHWAY 1.txt',header=None,sep="\t",on_bad_lines='skip')
k = [i.split(',') for i in list(drug_prior[2])]
target_genes = []
for i in k:
    target_genes += i
target_genes = [i.strip() for i in target_genes]
pathway_gene = pd.read_csv("pathway.Tingyi_gene.tsv",sep='\t')
drug_gene_interaction = pd.read_csv("interactions_laetst.tsv",sep='\t')
drug_target_df = pd.read_csv("gene.max_interaction_score.anti_neoplastic_or_immunotherapy.known_target_added.txt", sep="\t",header=None)
gene_important = np.unique(list(pathway_gene['ALDH2'])+list(drug_target_df[0])+
                           list(gene_filtered_var.columns)+list(mutation_avail_filter.columns)+target_genes)

pathway_gene = pd.read_csv("pathway.Tingyi_gene.tsv",sep='\t',header=None)
pathway_names = np.unique(list(pathway_gene[0]))
pathway_gene.set_index(0, inplace=True)
target_path_way = pd.read_csv('drug.pathway_with_target_gene.NES_rank.txt',sep='\t',header=None)
target_drug_name = list(target_path_way[0])
target_drug_pathway = list(target_path_way[1])
gene_set = {}
for i in pathway_names:
    pathway_gene_set = list(pathway_gene.loc[i][1])
    gene_set[i] = pathway_gene_set


with open('gene_embedding_important.npy', 'rb') as f:
    gene_embeddings = np.load(f)


GDSC_validate_drugs = ['PF-562271','QUIZARTINIB','FORETINIB','DABRAFENIB','SELUMETINIB','MASITINIB','FR-180204',
                       'GEFITINIB','AXITINIB','PALBOCICLIB','AFATINIB','JW-7-24-1','OSIMERTINIB','DASATINIB','BOSUTINIB',
                       'KU-55933','PONATINIB','CCT007093','WZ4003','TAMOXIFEN','IBRUTINIB','AZD3759','QL-XI-92','LESTAURTINIB',
                       'AZD6738','CAMPTOTHECIN','PRT062607','OSIMERTINIB','IPA-3','AFATINIB','VE-822','PFI-1','PAZOPANIB','QL-X-138',
                       'OSI-930','VISMODEGIB','AMUVATINIB','AZD4547','GW-2580','LFM-A13','MIRA-1','CP466722','SAVOLITINIB',
                       'SN-38','GSK690693','RUXOLITINIB','ALECTINIB','PELITINIB','AZ960','AST-1306','SERDEMETAN','PHA-665752','sulfatinib',
                      'NINTEDANIB','AZD4547']

GDSC_remove = ['PF-562271', 'QUIZARTINIB', 'SELUMETINIB', 'MASITINIB', 'FR-180204', 'JW-7-24-1', 
                'BOSUTINIB', 'PONATINIB', 'CCT007093', 'QL-XI-92', 'LESTAURTINIB', 'IPA-3', 'PFI-1', 'PAZOPANIB', 
                'QL-X-138', 'OSI-930', 'VISMODEGIB', 'AMUVATINIB', 'GW-2580', 'LFM-A13', 'CP466722', 'SN-38', 'GSK690693', 
                'ALECTINIB', 'PELITINIB', 'AST-1306', 'SERDEMETAN', 'PHA-665752', 'sulfatinib', 'NINTEDANIB']

GDSC_validate_drugs_remove = [i for i in GDSC_validate_drugs if not i in GDSC_remove]

lll = pd.read_csv('GDSC2_AUC_smile.csv')
lll.set_index('DRUG_NAME',inplace=True)
lll = lll.loc[GDSC_validate_drugs_remove]

lll.reset_index(inplace=True)

DepMap_id = pd.read_csv('/project/DPDS/Xiao_lab/shared/tingyi/GDSC_data/DepMap_2018q3_celllines.csv')
DepMap_id = DepMap_id.rename(columns={"Broad_ID":"ARXSPAN_ID"})
merge_gdsc_id = pd.merge(DepMap_id,lll,on='ARXSPAN_ID')

cell_line_remove = ['BDCM_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BL70_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'CAKI2_KIDNEY', 
'CAL12T_LUNG', 'CALU1_LUNG', 'CI1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'COLO201_LARGE_INTESTINE', 
'COLO320_LARGE_INTESTINE', 'COLO699_LUNG', 'COLO741_SKIN', 'COV318_OVARY', 'COV504_OVARY', 'DV90_LUNG', 
'EB1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EFE184_ENDOMETRIUM', 'F36P_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'GCIY_STOMACH', 'HCC2935_LUNG', 'HCC4006_LUNG', 'HCC56_LARGE_INTESTINE', 'HEC151_ENDOMETRIUM', 'HEC1A_ENDOMETRIUM', 
'HEC1B_ENDOMETRIUM', 'HEC251_ENDOMETRIUM', 'HEC265_ENDOMETRIUM', 'HEC59_ENDOMETRIUM', 'HEC6_ENDOMETRIUM', 
'HEL9217_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HEPG2_LIVER', 'HEYA8_OVARY', 'HLF_LIVER', 'HMC18_BREAST', 'HMCB_SKIN', 
'HS294T_SKIN', 'HS683_CENTRAL_NERVOUS_SYSTEM', 'HS695T_SKIN', 'HS729_SOFT_TISSUE', 'HS852T_SKIN', 'HS936T_SKIN', 
'HS939T_SKIN', 'HS944T_SKIN', 'HT_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HUT78_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'IGR39_SKIN', 'IMR32_AUTONOMIC_GANGLIA', 'IPC298_SKIN', 'ISHIKAWAHERAKLIO02ER_ENDOMETRIUM', 'ISTMES2_PLEURA', 
'JHH5_LIVER', 'JHOS2_OVARY', 'JHOS4_OVARY', 'JHUEM2_ENDOMETRIUM', 'JM1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'JMSU1_URINARY_TRACT', 'K029AX_SKIN', 'KASUMI2_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'KE39_STOMACH', 
'KHM1B_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'KMBC2_URINARY_TRACT', 'KMM1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'KMRC2_KIDNEY', 'KMS11_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'KMS26_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'KMS34_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'KNS42_CENTRAL_NERVOUS_SYSTEM', 'KNS60_CENTRAL_NERVOUS_SYSTEM', 
'KNS81_CENTRAL_NERVOUS_SYSTEM', 'KO52_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'KPNSI9S_AUTONOMIC_GANGLIA', 
'KYSE30_OESOPHAGUS', 'L33_PANCREAS', 'LU99_LUNG', 'LUDLU1_LUNG', 'MALME3M_SKIN', 'MC116_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'MCAS_OVARY', 'MDAMB435S_SKIN', 'MEC1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'MELHO_SKIN', 'MINO_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'MJ_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'MKN74_STOMACH', 'MONOMAC1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'MORCPR_LUNG', 
'NCIH1184_LUNG', 'NCIH1339_LUNG', 'NCIH1373_LUNG', 'NCIH2030_LUNG', 'NCIH2172_LUNG', 'NCIH2286_LUNG', 'NCIH2444_LUNG', 
'NCIH322_LUNG', 'NCIH3255_LUNG', 'NCIH460_LUNG', 'NCIH647_LUNG', 'NCO2_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'OC316_OVARY', 
'ONCODG1_OVARY', 'OVMANA_OVARY', 'OVSAHO_OVARY', 'P3HR1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'PFEIFFER_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'PK1_PANCREAS', 'PK45H_PANCREAS', 'PK59_PANCREAS', 
'PLCPRF5_LIVER', 'RERFLCAI_LUNG', 'RERFLCMS_LUNG', 'RVH421_SKIN', 'SCABER_URINARY_TRACT', 
'SCC25_UPPER_AERODIGESTIVE_TRACT', 'SF126_CENTRAL_NERVOUS_SYSTEM', 'SF295_CENTRAL_NERVOUS_SYSTEM', 
'SH10TC_STOMACH', 'SIGM5_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'SKBR3_BREAST', 'SKLU1_LUNG', 'SKMEL30_SKIN', 
'SKMEL31_SKIN', 'SKNBE2_AUTONOMIC_GANGLIA', 'SNU475_LIVER', 'SNUC2A_LARGE_INTESTINE', 'SQ1_LUNG', 
'SUDHL4_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'SW1353_BONE', 'SW403_LARGE_INTESTINE', 'SW480_LARGE_INTESTINE',
 'SW579_THYROID', 'T24_URINARY_TRACT', 'T3M10_LUNG', 'TE11_OESOPHAGUS', 'TE617T_SOFT_TISSUE', 'TEN_ENDOMETRIUM', 
 'TOLEDO_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'U937_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'VMRCRCW_KIDNEY', 'WM1799_SKIN', 
 'WM2664_SKIN', 'WM88_SKIN', 'WM983B_SKIN', 'YKG1_CENTRAL_NERVOUS_SYSTEM', 'ZR751_BREAST']

cell_line_name_avail_ = [i for i in cell_line_name_avail if not i in cell_line_remove]
merge_gdsc_id.set_index('CCLE_Name',inplace=True)

test_cell_line_name = cell_line_name_avail_[0:200]

merge_gdsc_id = merge_gdsc_id.loc[test_cell_line_name]



input_gene_exp_one_hot_val = []
input_drug_one_hot_val = []
input_drug_response_val = []
input_rel_distance_val = []
input_smile_seq_val = []
input_gene_prior_val = []
input_cell_line_name_val = []
input_drug_name_val = []
input_interpret_smile_val = []
for i in test_cell_line_name:
    index = 0
    print(i)
    for j in GDSC_validate_drugs_remove:
        try:
            drug_smile_single = merge_gdsc_id.loc[i][merge_gdsc_id.loc[i]['DRUG_NAME']==j]['canonical_smile'][0]
            #drug_smile_single = CCLE_drug_smiles[index]
            rel_distance_ = generate_rel_dist_matrix(drug_smile_single)
            interpret_smile = generate_interpret_smile(drug_smile_single)[0]
            input_interpret_smile_val.append(interpret_smile)
            input_drug_name_val.append(j)
            #input_rel_distance.append(rel_distance_)
            input_smile_seq_val.append(drug_smile_single)
            input_drug_response_val.append(merge_gdsc_id.loc[i][merge_gdsc_id.loc[i]['DRUG_NAME']==j]['AUC_PUBLISHED'][0])
            input_cell_line_name_val.append(i)
        except:
            continue
    index+=1


smile_length = 100
rel_distance_batch_val = [generate_rel_dist_matrix(x) for x in input_smile_seq_val]
drug_rel_position_chunk_val = []
drug_smile_length_chunk_val = []
drug_atom_one_hot_chunk_val = []
gene_mutation_chunk_val = []
gene_prior_chunk_val = []
edge_type_matrix_chunk_val = []
gene_expression_chunk_val = []
for rel_distance_ in rel_distance_batch_val:
    shape = rel_distance_.shape[0]
    drug_rel_position = tf.cast(tf.gather(P[0], tf.cast(rel_distance_,tf.int32), axis=0), tf.float32)
    concat_left = tf.cast(tf.zeros((smile_length-shape,shape,60)), tf.float32)
    concat_right = tf.cast(tf.zeros((smile_length,smile_length-shape,60)), tf.float32)
    drug_rel_position = tf.concat((drug_rel_position,concat_left),axis=0)
    drug_rel_position = tf.concat((drug_rel_position,concat_right),axis=1)
    drug_rel_position_chunk_val.append(drug_rel_position)
drug_rel_position_chunk_val = tf.stack(drug_rel_position_chunk_val)

for interpret_smile in input_interpret_smile_val:
    input_drug_atom_names = tf.constant(list(interpret_smile))
    input_drug_atom_index = string_lookup(input_drug_atom_names)-1
    input_drug_atom_one_hot = layer_one_hot(input_drug_atom_index)
    shape_drug_miss = input_drug_atom_one_hot.shape[0]
    concat_right = tf.zeros((smile_length-shape_drug_miss,8))
    input_drug_atom_one_hot = tf.concat((input_drug_atom_one_hot,concat_right),axis=0)
    drug_smile_length_chunk_val.append(shape_drug_miss)
    drug_atom_one_hot_chunk_val.append(input_drug_atom_one_hot)
drug_smile_length_chunk_val = np.array(drug_smile_length_chunk_val)
drug_atom_one_hot_chunk_val = tf.stack(drug_atom_one_hot_chunk_val)

for smile_seq in input_smile_seq_val:
    edge_type_matrix = get_drug_edge_type(smile_seq)
    shape = edge_type_matrix.shape[0]
    edge_type_matrix = tf.gather(edge_type_dict,tf.cast(edge_type_matrix,tf.int16),axis=0)
    #drug_rel_position = tf.cast(tf.gather(P[0], tf.cast(rel_distance_,tf.int32), axis=0), tf.float32)
    concat_left = tf.zeros((smile_length-shape,shape,5))
    concat_right = tf.zeros((smile_length,smile_length-shape,5))
    edge_type_matrix = tf.concat((edge_type_matrix,concat_left),axis=0)
    edge_type_matrix = tf.concat((edge_type_matrix,concat_right),axis=1)
    edge_type_matrix_chunk_val.append(edge_type_matrix)
edge_type_matrix_chunk_val = tf.stack(edge_type_matrix_chunk_val)

for cell_line_ in input_cell_line_name_val:
    gene_expression_singlecelline = continuous_gene_df_filter.loc[cell_line_]
    gene_expression_chunk_val.append(gene_expression_singlecelline)
    gene_mutation_singlecelline = mutation_whole.loc[cell_line_]
    gene_mutation_chunk_val.append(gene_mutation_singlecelline)

gene_prior_chunk_val = tf.stack(input_gene_prior_val)
gene_expression_chunk_val = tf.stack(gene_expression_chunk_val)
gene_expression_bin_chunk_val = tf.gather(gene_expression_bin_dict,tf.cast(gene_expression_chunk_val,tf.int16),axis=0)
gene_mutation_chunk_val = tf.stack(gene_mutation_chunk_val)
gene_mutation_bin_chunk_val = tf.gather(gene_mutation_dict,tf.cast(gene_mutation_chunk_val,tf.int16),axis=0)

batch_shape_val = gene_prior_chunk_val.shape[0]
mask_val = tf.range(start=0, limit=100, dtype=tf.float32)
mask_val = tf.broadcast_to(tf.expand_dims(mask_val,axis=0),shape=[batch_shape_val,100])
mask_val = tf.reshape(mask_val, shape=(batch_shape_val*100))
mask_val = mask_val < tf.cast(tf.repeat(drug_smile_length_chunk_val,repeats=100),tf.float32)
mask_val = tf.where(mask_val,1,0)
mask_val = tf.reshape(mask_val, shape=(batch_shape_val,100))
mask_val = tf.expand_dims(mask_val, axis=-1)





