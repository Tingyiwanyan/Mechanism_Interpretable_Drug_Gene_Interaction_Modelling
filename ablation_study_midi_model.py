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

gene_name_avail_geneformer = list(np.load('gene_names.npy'))

gene_expression_whole_avail = gene_expression.loc[cell_line_name_avail]
disc_gene_total = []
continuous_gene_total = []
for name in cell_line_name_avail:
    #print(name)
    max_value = np.max(np.array(gene_expression.loc[name]))
    #min_value = np.min(np.array(gene_expression.loc[name]))
    continuous_gene = normalize_min_max(gene_expression_whole_avail.loc[name])
    continuous_gene_total.append(continuous_gene)
    bin_value = max_value/5
    #bin_value = max_value/4
    #Dis = tf.keras.layers.Discretization(bin_boundaries=[0, bin_value,2*bin_value, 3*bin_value ],epsilon=0.001)
    Dis = tf.keras.layers.Discretization(bin_boundaries=[bin_value,2*bin_value,3*bin_value],epsilon=0.001)
    #Dis.adapt(np.array(gene_expression_whole_avail))
    disc_gene_ = Dis(np.array(gene_expression_whole_avail.loc[name]))
    disc_gene_total.append(disc_gene_)
disc_gene_total = tf.stack(disc_gene_total)
disc_gene_df = pd.DataFrame(disc_gene_total, index=cell_line_name_avail)
disc_gene_df.columns = list(gene_expression.columns)
disc_gene_df_filter = disc_gene_df[gene_name_avail_geneformer]

continuous_gene_total = tf.stack(continuous_gene_total)
continuous_gene_df = pd.DataFrame(continuous_gene_total, index=cell_line_name_avail)
continuous_gene_df.columns = list(gene_expression.columns)
continuous_gene_df_filter = continuous_gene_df[gene_name_avail_geneformer]

avail_mutation_list = disc_gene_df_filter.columns.intersection(mutation_avail_filter.columns)
mutation_avail_filter = mutation_avail_filter[avail_mutation_list]
mutation_whole = np.zeros(np.array(disc_gene_df_filter).shape)
mutation_whole = pd.DataFrame(mutation_whole, index=cell_line_name_avail)
mutation_whole.columns = list(disc_gene_df_filter.columns)
mutation_whole[mutation_avail_filter.columns] = mutation_avail_filter

prior_drug_information_total = pd.read_csv('prior_drug_gene_target_info.csv')

string_lookup = tf.keras.layers.StringLookup(vocabulary=vocabulary_drug)
layer_one_hot = tf.keras.layers.CategoryEncoding(num_tokens=8, output_mode="one_hot")
check_prior_drug_gene = pd.read_excel('drug.Tingyi_gene.sorted_by_interaction_score 2.xlsx')
check_prior_drug_gene = check_prior_drug_gene.replace([np.nan], 0)
input_gene_prior_convert = []
input_rel_distance_convert = []
input_smile_seq_convert = []
input_drug_name_convert = []
input_interpret_smile_convert = []
input_prior_gene_names = []
for i in np.unique(prior_drug_information_total['original_drug_smile']):
    #index = 0
    #print(i)
    flag = 0
    single_drug_df =prior_drug_information_total.set_index('original_drug_smile').loc[i]
    try:
        single_drug_df['rank_score'] = single_drug_df['interaction_score'].rank(ascending = 0)
        single_drug_df = single_drug_df.set_index('rank_score').sort_index()
        prior_genes = list(single_drug_df['targeted_genes'])[0:10]
    except:
        prior_genes = single_drug_df['targeted_genes']
        flag = 1
    #print(prior_genes)
    all_one_vector = np.zeros(6144)
    all_zero_vector = np.ones(6144)*-1e7
    indexes = []
    if prior_genes == ['GLRB', 'KCNK2', 'GABRE', 'GABRP', 'GABRB3', 'GABRA3', 'GABRA2']:
        print(i)
    if flag == 0:
        for each_gene in prior_genes:
            try:
                index = np.where(np.array(gene_name_avail_geneformer)==each_gene)[0][0]
                indexes.append(index)
            except:
                continue
    else:
        index = np.where(np.array(gene_name_avail_geneformer)==prior_genes)[0][0]
        indexes.append(index)
    if indexes == []:
        print('Im here')
        print(prior_genes)
        print(single_drug_df)
        input_gene_prior_convert.append(all_one_vector)
    else:
        for each_gene_index in indexes:
            all_zero_vector[each_gene_index] = 0
        input_gene_prior_convert.append(all_zero_vector)
    
    drug_smile_single = i#CCLE_drug_smiles[index]
    #print(drug_smile_single)
    rel_distance_ = generate_rel_dist_matrix(drug_smile_single)
    interpret_smile = generate_interpret_smile(drug_smile_single)[0]
    input_interpret_smile_convert.append(interpret_smile)
    try:
        drug_name_convert_single = prior_drug_information_total.set_index('original_drug_smile').loc[i].iloc[0]['drug_names']
    except:
        drug_name_convert_single = prior_drug_information_total.set_index('original_drug_smile').loc[i]['drug_names']
    #print(drug_name_convert_single)
    input_drug_name_convert.append(drug_name_convert_single)
    #input_rel_distance.append(rel_distance_)
    input_smile_seq_convert.append(drug_smile_single)
    input_rel_distance_convert.append(rel_distance_)
    input_prior_gene_names.append(prior_genes)


cell_line_drug.set_index('Cell_line_Name',inplace=True)

df_drug_smile = pd.DataFrame(list(zip(drug_names,CCLE_drug_smiles)),
                                 columns=['drug_name','drug_smiles'])
df_drug_smile.set_index('drug_name',inplace =True)
cell_line_name_val = cell_line_name_avail[0:60]
input_gene_exp_one_hot_val = []
input_drug_one_hot_val = []
input_drug_response_val = []
input_rel_distance_val = []
input_smile_seq_val = []
input_gene_prior_val = []
input_cell_line_name_val = []
input_drug_name_val = []
input_interpret_smile_val = []
for i in cell_line_name_val:
#for i in leukemia_cell_lines:
    index = 0
    print(i)
    for j in drug_names:
        #if not j == 'nilotinib':
            #continue
        #print(j)
        if not np.isnan(cell_line_drug.loc[i][j]):
            all_one_vector = np.zeros(6144)
            all_zero_vector = np.ones(6144)*-1e7
            if not np.isnan(cell_line_drug.loc[i][j]):
                try:
                    #prior_genes = drug_prior.loc[j][2].split(',')
                    prior_genes = check_prior_drug_gene[j]
                    indexes = []
                    for each_gene in prior_genes:
                        if not each_gene == 0:
                            try:
                                #index = np.where(np.array(gene_name_avail_geneformer)==each_gene.strip())[0][0]
                                index = np.where(np.array(gene_name_avail_geneformer)==each_gene)[0][0]
                                indexes.append(index)
                            except:
                                continue
                    if indexes == []:
                        #print("no drug target")
                        #print(j)
                        input_gene_prior_val.append(all_one_vector)
                    else:
                        for each_gene_index in indexes:
                            all_zero_vector[each_gene_index] = 0
                        input_gene_prior_val.append(all_zero_vector)
                except:
                    input_gene_prior_val.append(all_one_vector)
            drug_smile_single = df_drug_smile.loc[j][0]
            #drug_smile_single = CCLE_drug_smiles[index]
            rel_distance_ = generate_rel_dist_matrix(drug_smile_single)
            interpret_smile = generate_interpret_smile(drug_smile_single)[0]
            input_interpret_smile_val.append(interpret_smile)
            input_drug_name_val.append(j)
            #input_rel_distance.append(rel_distance_)
            input_smile_seq_val.append(drug_smile_single)
            input_drug_response_val.append(cell_line_drug.loc[i][j])
            input_cell_line_name_val.append(i)
        index+=1

"""
Create validation input set
"""
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

class drug_transformer_():
    """
    Implement the drug transformer model architecture
    """
    def __init__(self, gene_embeddings):#, relative_pos_enc_lookup=None):

        self.input_gene_embeddings = gene_embeddings

        self.masked_softmax_ = masked_softmax()
        self.masked_softmax_2 = masked_softmax()
        self.masked_softmax_deco_self = masked_softmax()
        self.masked_softmax_deco_self2 = masked_softmax()
        self.masked_softmax_deco_cross = masked_softmax()
        self.masked_softmax_deco_cross2 = masked_softmax()

        self.feature_selection = feature_selection_layer_global_drug()

        """
        global decoder
        """
        self.decoder_global_1 = decoder_cross_block(30)#, if_select_feature_=True)
        self.decoder_global_2 = decoder_cross_block(30, if_select_feature_=True)
        self.decoder_global_3 = decoder_cross_block(30, if_select_feature_=True)

        self.encoder_1 = encoder_block(60,130)

        self.encoder_2 = encoder_block(60,130)
        self.encoder_3 = encoder_block(30,130)
        self.feature_select_cross = feature_selection_cross_block(60, if_select_feature_=True)

        """
        1st head attention
        """
        self.dotproductattention1 = dotproductattention(30)

        #self.dotproductattention_deco = dotproductattention_column(30)

        self.dotproductattention_deco_cross = dotproductattention(30)

        self.decoder_cross_1 = decoder_cross_block(60)

        """
        2nd head attention
        """
        self.dotproductattention2 = dotproductattention(15)

        self.dotproductattention_deco2 = dotproductattention(10)

        self.dotproductattention_deco_cross2 = dotproductattention(10)

        self.decoder_cross_2 = decoder_cross_block(60)


        """
        3rd head attention
        """
        self.dotproductattention3 = dotproductattention(10)

        self.dotproductattention_deco3 = dotproductattention(10)

        self.dotproductattention_deco_cross3 = dotproductattention(10)

        self.decoder_cross_3 = decoder_cross_block(10)

        self.decoder_cross_4 = decoder_cross_block(30)
        self.decoder_cross_5 = decoder_cross_block(30)
        self.decoder_cross_6 = decoder_cross_block(30)



        #self.att_embedding = attention_embedding()
        self.r_connection = residual_connection()
        self.r_connection_gene_emb = residual_connection()
        self.r_connection_gene_mutate = residual_connection()
        self.r_connection_multi_deco_gene = residual_connection()
        self.r_connection_feature = residual_connection()

        self.dense_0 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_0")

        self.dense_1 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_1")

        self.dense_2 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_2")

        self.dense_3 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_3")

        self.dense_4 = tf.keras.layers.Dense(30, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_4")

        self.dense_8 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_8")

        self.dense_5 = tf.keras.layers.Dense(1, kernel_initializer=initializers.RandomNormal(seed=42),
                                             #kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_5")

        self.dense_6 = tf.keras.layers.Dense(1, activation='relu', 
                                             kernel_initializer=initializers.RandomNormal(seed=42),
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_6")


        self.dense_9 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_9")

        self.dense_12 = tf.keras.layers.Dense(30, kernel_initializer=initializers.RandomNormal(seed=42),
                                              activation='relu',
                                              kernel_regularizer=regularizers.L2(1e-4),
                                              bias_initializer=initializers.Zeros(), name="dense_12")

        self.dense_13 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                              activation='relu',
                                              kernel_regularizer=regularizers.L2(1e-4),
                                              bias_initializer=initializers.Zeros(), name="dense_13")

        self.dense_14 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                              activation='relu',
                                              kernel_regularizer=regularizers.L2(1e-4),
                                              bias_initializer=initializers.Zeros(), name="dense_14")

        self.dense_15 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                              activation='relu',
                                              kernel_regularizer=regularizers.L2(1e-4),
                                              bias_initializer=initializers.Zeros(), name="dense_15")

        self.dense_16 = tf.keras.layers.Dense(60, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_16")

        self.dense_17 = tf.keras.layers.Dense(1, kernel_initializer=initializers.RandomNormal(seed=42),
                                             #kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_17")

        self.dense_18 = tf.keras.layers.Dense(30, kernel_initializer=initializers.RandomNormal(seed=42),
                                             activation='relu',
                                             kernel_regularizer=regularizers.L2(1e-4),
                                             bias_initializer=initializers.Zeros(), name="dense_18")

        self.pos_encoding = positionalencoding(30,130)

        self.pos_encoding_gene = positionalencoding(30, 6144)
        self.flattern_enco = tf.keras.layers.Flatten()
        self.flattern_deco = tf.keras.layers.Flatten()
        self.flattern_score = tf.keras.layers.Flatten()
        self.flattern_global = tf.keras.layers.Flatten()
        self.flattern_global_ = tf.keras.layers.Flatten()


        self.dotproductattention = dotproductattention(768)

        self.kernel_value = tf.keras.layers.Dense(768, kernel_initializer=initializers.RandomNormal(seed=42),
                                                  kernel_regularizer=regularizers.L2(1e-4),
                                                  bias_initializer=initializers.Zeros())

    def model_construction_midi(self, if_mutation=None):
        """
        construct the transformer model
        """
        X_input = Input((100, 8))
        Y_input = Input((6144, 1))
        gene_mutation_input = Input((6144, 2))
        rel_position_embedding = Input((100,100,60))
        edge_type_embedding = Input((100,100,5))
        #rel_position_embedding_origin = Input((80,80,60))
        enc_valid_lens_ = Input(())
        mask_input = Input((100,1))

        shape_input = tf.shape(X_input)
        gene_embedding = self.input_gene_embeddings
        gene_embedding = tf.expand_dims(gene_embedding, axis=0)
        gene_embedding = tf.broadcast_to(gene_embedding, [shape_input[0], gene_embedding.shape[1], gene_embedding.shape[-1]])

        gene_embedding = tf.math.l2_normalize(self.dense_3(gene_embedding),axis=-1)
        #gene_embedding = self.dense_3(gene_embedding)

        #rel_position_embedding_ = tf.math.l2_normalize(self.dense_13(rel_position_embedding), axis = -1)
        edge_type_embedding_ = tf.math.l2_normalize(self.dense_8(edge_type_embedding),axis=-1)

        X = self.dense_0(X_input)
        #X = self.pos_encoding(X)
        X, att, score = self.encoder_1(X, enc_valid_lens=enc_valid_lens_, 
                                #relative_pos_enc=self.relative_pos_enc_lookup,
                                relative_pos_enc=rel_position_embedding,
                                edge_type_enc = edge_type_embedding_,
                                #relative_pos_origin_ = rel_position_embedding_origin,
                                if_sparse_max=False)

        X, att, score = self.encoder_2(X, enc_valid_lens=enc_valid_lens_, 
                                #relative_pos_enc=self.relative_pos_enc_lookup,
                                relative_pos_enc=rel_position_embedding,
                                edge_type_enc = edge_type_embedding_,
                                #relative_pos_origin_ = rel_position_embedding_origin,
                                if_sparse_max=False)
        #X_enc_2, att = self.encoder_2(X, enc_valid_lens=enc_valid_lens_,
                                     #relative_pos_enc=self.relative_pos_enc_lookup)
        #X_enc_3, att = self.encoder_3(X, enc_valid_lens=enc_valid_lens_)
        #X = tf.concat([X_enc_1, X_enc_2],axis=-1)

        X = self.dense_1(X)

        shape_x = tf.shape(X)
        #mask = tf.expand_dims(mask_input, axis=-1)
        mask = tf.cast(tf.broadcast_to(mask_input, shape=shape_x),tf.float32)
        X = tf.multiply(mask,X)

        
        X_global = self.flattern_global(X)
        #X_global = tf.reduce_sum(X, axis=1)
        #X_global = tf.math.divide(X_global, tf.expand_dims(enc_valid_lens_,axis=-1))
        X_global = tf.expand_dims(X_global, axis=1)
        X_global = self.dense_9(X_global)

        """
        self-attention for the decoder
        """
        Y = tf.math.l2_normalize(self.dense_2(Y_input),axis=-1)
        #Y = self.dense_2(Y_input)
        #Y = tf.math.l2_normalize(tf.concat([gene_embedding, Y],axis=-1),axis=-1)
        #Y = self.r_connection_gene_emb(Y, gene_embedding)
        Y = tf.math.add(Y, gene_embedding)
        #Y = tf.concat([Y,gene_embedding], axis=-1)

        if not if_mutation == None:
            Y_gene_mutate = tf.math.l2_normalize(self.dense_14(gene_mutation_input),axis=-1)
            #Y = tf.math.l2_normalize(tf.concat([Y, Y_gene_mutate],axis=-1),axis=-1)
            #Y = self.r_connection_gene_mutate(Y, Y_gene_mutate)
            Y = tf.math.add(Y, Y_gene_mutate)
            #Y = tf.concat([Y, Y_gene_mutate], axis=-1)
        #Y = self.pos_encoding_gene(Y)

        Y = self.dense_16(Y)

        """
        cross attention for the decoder
        """

        X_global, att_score_global1, Y_value, score_cross = self.decoder_global_1(X_global, gene_embedding, if_sparse_max=False)#, if_select_feature_=None)
        X_global_, att_score_global2, Y_key, score_cross_global = self.decoder_global_2(X_global, Y_value, if_sparse_max=False, if_select_feature_=True)
        #X_global3, att_score_global3, Y_key3 = self.decoder_global_3(X_global, Y, if_sparse_max=True, if_select_feature_=True)

        #X_global1, att_score_global1 = self.decoder_global_1(X_global, Y, if_sparse_max=True)
        #X_global2, att_score_global2 = self.decoder_global_2(X_global, Y, if_sparse_max=True)
        #X_global3, att_score_global3 = self.decoder_global_3(X_global, Y, if_sparse_max=True)

        #Y = tf.concat([Y,Y_key], axis=-1)
        #Y = self.dense_16(Y)

        #X_global = X_global1
        #att_score_global1 = tf.transpose(att_score_global1, perm=[0,2,1])
        att_score_global2 = tf.transpose(att_score_global2, perm=[0,2,1])
        #att_score_global3 = tf.transpose(att_score_global3, perm=[0,2,1])

        #X_global_att = tf.broadcast_to(X_global, shape=[shape_input[0], Y_key.shape[1], Y_key.shape[-1]])
        #Y_key = tf.math.multiply(att_score_global2, Y_key)s
        #Y_key = tf.math.add(X_global_att, Y_key)

        #X_global, att_score_global2, Y_key = self.decoder_global_2(X_global, Y, if_sparse_max=False, if_select_feature_=True)
        #Y = self.dense_6(Y)
        #Y_key2 = self.dense_6(Y_key2)
        #Y_key3 = self.dense_6(Y_key3)
        #att_score_global2 = tf.transpose(att_score_global2, perm=[0,2,1])
        Y_global = tf.math.multiply(att_score_global2, Y)

        Y = Y_global
        X_global = self.flattern_global_(X_global_)
        #X_global = tf.math.l2_normalize(X_global, axis=-1)
        X_global = self.dense_17(X_global)
        Y = self.flattern_deco(Y)
        Y = tf.math.l2_normalize(Y, axis=-1)
        Y = self.dense_18(Y)
        #Y = tf.math.l2_normalize(Y, axis=-1)
        #Y = tf.concat([X_global, Y], axis=-1)   
        Y = self.dense_5(Y)
        Y_predict = tf.math.add(Y, X_global)


        self.model = Model(inputs=(X_input, Y_input, enc_valid_lens_, rel_position_embedding, edge_type_embedding, gene_mutation_input, mask_input), \
            outputs=[Y_predict, score_cross_global, X_global, Y, gene_embedding, X_global_, att_score_global2, Y_global])
        #self.model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

        return self.model

k = drug_transformer_(gene_embeddings)#, relative_pos_enc_lookup=relative_pos_embedding)
#midi_model_no_supervised_contrast = k.model_construction_midi(if_mutation=True)
#midi_model_no_supervised_contrast.summary()


#midi_model_no_supervised_contrast.load_weights('/project/DPDS/Xiao_lab/shared/tingyi/drug_sensitivity_prediction/Drug_response/BIB_revision/midi_no_supervise_contrast.weights.h5')


midi_model_no_self_supervised_contrast = k.model_construction_midi(if_mutation=True)
midi_model_no_self_supervised_contrast.summary()


midi_model_no_self_supervised_contrast.load_weights('/project/DPDS/Xiao_lab/shared/tingyi/drug_sensitivity_prediction/Drug_response/BIB_revision/midi_no_self_supervised.weights.h5')

#midi_simple_concat.summary()

#midi_simple_concat.load_weights('midi_simple_concat.weights.h5')

prediction_val_1 = midi_model_no_self_supervised_contrast((drug_atom_one_hot_chunk_val[0:400], gene_expression_chunk_val[0:400], 
                                         drug_smile_length_chunk_val[0:400], drug_rel_position_chunk_val[0:400], 
                                         edge_type_matrix_chunk_val[0:400], gene_mutation_bin_chunk_val[0:400],mask_val[0:400]))[0][:,0]

prediction_val_2 = midi_model_no_self_supervised_contrast((drug_atom_one_hot_chunk_val[400:800], gene_expression_chunk_val[400:800], 
                                         drug_smile_length_chunk_val[400:800], drug_rel_position_chunk_val[400:800], 
                                         edge_type_matrix_chunk_val[400:800], gene_mutation_bin_chunk_val[400:800],mask_val[400:800]))[0][:,0]

prediction_val_3 = midi_model_no_self_supervised_contrast((drug_atom_one_hot_chunk_val[800:1200], gene_expression_chunk_val[800:1200], 
                                         drug_smile_length_chunk_val[800:1200], drug_rel_position_chunk_val[800:1200], 
                                         edge_type_matrix_chunk_val[800:1200], gene_mutation_bin_chunk_val[800:1200],mask_val[800:1200]))[0][:,0]

prediction_val_4 = midi_model_no_self_supervised_contrast((drug_atom_one_hot_chunk_val[1200:], gene_expression_chunk_val[1200:], 
                                         drug_smile_length_chunk_val[1200:], drug_rel_position_chunk_val[1200:], 
                                         edge_type_matrix_chunk_val[1200:], gene_mutation_bin_chunk_val[1200:],mask_val[1200:]))[0][:,0]


prediction_val = np.concatenate([prediction_val_1,prediction_val_2,prediction_val_3,prediction_val_4])



acc = scipy.stats.pearsonr(np.array(input_drug_response_val),prediction_val)[0]

#np.save('BIB_revision/midi_simple_concat.npy', prediction_val)
#np.save('BIB_revision/input_drug_response_val.npy', np.array(input_drug_response_val))

#np.save('BIB_revision/midi_no_supervise_contrast.npy', prediction_val)




