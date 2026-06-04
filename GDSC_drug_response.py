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




