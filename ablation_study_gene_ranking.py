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


#with open('gene_embedding_important.npy', 'rb') as f:
    #gene_embeddings = np.load(f)

gene_name_avail_geneformer = list(np.load('gene_names.npy'))

df_gene_embedding_one_hot = pd.DataFrame({'numbers': range(6144)})
encoder_gene_embedding = BinaryEncoder(cols=['numbers'])
df_binary_gene_embedding = encoder_gene_embedding.fit_transform(df_gene_embedding_one_hot)

gene_embeddings = np.array(df_binary_gene_embedding)

k = drug_transformer_(gene_embeddings)
midi_model_binary_gene_embedding = k.model_construction_midi(if_mutation=True)
midi_model_binary_gene_embedding.load_weights('/project/DPDS/Xiao_lab/shared/tingyi/drug_sensitivity_prediction/Drug_response/BIB_revision/midi_binary_gene_embedding.weights.h5')



GDSC_validate_drugs = ['PF-562271','QUIZARTINIB','FORETINIB','DABRAFENIB','SELUMETINIB','MASITINIB','FR-180204',
                       'GEFITINIB','AXITINIB','PALBOCICLIB','AFATINIB','JW-7-24-1','OSIMERTINIB','DASATINIB','BOSUTINIB',
                       'KU-55933','PONATINIB','CCT007093','WZ4003','TAMOXIFEN','IBRUTINIB','AZD3759','QL-XI-92','LESTAURTINIB',
                       'AZD6738','CAMPTOTHECIN','PRT062607','OSIMERTINIB','IPA-3','AFATINIB','VE-822','PFI-1','PAZOPANIB','QL-X-138',
                       'OSI-930','VISMODEGIB','AMUVATINIB','AZD4547','GW-2580','LFM-A13','MIRA-1','CP466722','SAVOLITINIB',
                       'SN-38','GSK690693','RUXOLITINIB','ALECTINIB','PELITINIB','AZ960','AST-1306','SERDEMETAN','PHA-665752','sulfatinib',
                      'NINTEDANIB','AZD4547']

TTD_validate_drugs = ['Intedanib','Ruxolitinib','Baricitinib','Estrone','Ospemifene','Sulfatinib','EXISULIND','CG-100649',
                      'Giredestrant','NPC-01','Cyclothiazide','Benzthiazide','Quinestrol','Toremifene','Imlunestrant','Epigallocatechin gallate',
                      'FPL-62064','ABT-761','CP-868596','Fidarestat','Rivoceranib','Brivanib','Tolebrutinib','ICP-022','RG7388',
                      'Benserazide','LC-150444','MK-3102','SYR-472','Gemigliptin','Mapracorat','CORT-125134','OSI-906','Tivantinib',
                      'Seocalcitol','Rolofylline','AC-170','Rivoglitazone','Balaglitazone','FARGLITAZAR','Leriglitazone','Ragaglitazar',
                      'TESAGLITAZAR','MURAGLITAZAR','Imiglitazar','Edotecarin','CQA 206-291','Sumanirole','esamisulpride','Blonanserin',
                      'Enzastaurin','Buparlisib','Tramiprosate','ICI 118,551', 'ABT-263','INCB24360', 'SHR0302','MIN-101', 'Tozadenant',
                      'Binodenoson','Apadenoson','CYTISINE','AMD-070','Viramidine', 'Darusentan','Clazosentan','Paltusotine','CG-100649',
                      'Efaproxyn','Sivelestat','Retosiban','Tamoxifen','Elacestrant','Estradiol','Danazol','Dienestrol','Promestriene',
                      'Clomifene','Mestranol','Cyclofenil','Estrogen','Masoprocol','Zileuton','Dabrafenib','Epalrestat','Avapritinib',
                      'Moclobemide','Tranylcypromine','Galantamine','Rivastigmine','Huperzine A','Axitinib','Pacritinib','Novolimus',
                      'Everolimus','Sirolimus','Acalabrutinib','Zanubrutinib','Pirtobrutinib','IPI-145','Aminocaproic Acid',
                      'Flucinolone Acetonide','Triamcinolone', 'Mometasone','Betamethasone Valerate','Methylprednisolone','Deflazacort',
                      'Betamethasone','Prednisolone','Flunisolide','Meprednisone','Prednisone','Dexamethasone','Hydrocortisone',
                      'Fluticasone','Fluorometholone','Budesonide','GW685698X','Mifepristone','Lorlatinib','Oxandrolone','Calcipotriol',
                      'Calcidiol','Ergocalciferol','Doxercalciferol','Dihydrotachysterol','Cholecalciferol','Paricalcitol','Calcitriol',
                      'Fluorouracil','FENBUFEN','Aminosalicylic Acid','Rasagiline','Phenelzine','Tranylcypromine','Linagliptin','Vildagliptin',
                      'Anagliptin','Sitagliptin','Sparsentan','Carbinoxamine','Cyproheptadine','Desloratadine','Dexbrompheniramine','Olopatadine',
                      'Phenindamine','Dimethindene','Bropheniramine','Mequitazine','Tripelennamine','Chlorpheniramine','Cetirizine','Pemirolast',
                      'Methdilazine','Azatadine','Hydroxyzine','Alcaftadine','Diphenylpyraline','Tofacitnib']

TTD_validate_drugs = list(np.unique(TTD_validate_drugs))


#df_TTD_drug_smile = pd.read_csv('df_TTD_drug_smile.csv')

df_valid_drug_smile_TTD = pd.read_csv('valid_drug_smile_TTD.csv')

TTD_validate_drugs = list(np.unique(list(df_valid_drug_smile_TTD['Drug_name_TTD'])))

df_valid_drug_smile_TTD.set_index("Drug_name_TTD",inplace=True)


TTD_validate_smiles = []
for i in TTD_validate_drugs:
    try:
        np.array(df_valid_drug_smile_TTD.loc[i]['Drug_smile_TTD']).shape[0] 
        smile_ = df_valid_drug_smile_TTD.loc[i]['Drug_smile_TTD'][0]
        TTD_validate_smiles.append(smile_)
    except:
        smile_ = df_valid_drug_smile_TTD.loc[i]['Drug_smile_TTD']
        print(i)
        print(smile_)
        TTD_validate_smiles.append(smile_)


#k = drug_transformer_(gene_embeddings)#, relative_pos_enc_lookup=relative_pos_embedding)
#model_midi = k.model_construction_midi(if_mutation=True)

df_data = pd.read_csv('df_sample_drug_response_data.csv')
drug_names = list(df_data['drug_name'])

continuous_gene_exp = pd.read_csv('continuous_gene_exp.csv')
continuous_gene_exp.rename(columns = {continuous_gene_exp.columns[0]:'cell_line_name'}, inplace=True)
continuous_gene_exp.set_index('cell_line_name', inplace=True)


mutation_gene = pd.read_csv('mutation_gene.csv')
mutation_gene.rename(columns = {mutation_gene.columns[0]:'cell_line_name'}, inplace=True)
mutation_gene.set_index('cell_line_name', inplace=True)
mutation_gene



from utils.utils import *
from utils.smile_rel_dist_interpreter import *
#batch_drug_names = valid_drug_TTD_[300:400]
#batch_smile_seq = valid_drug_smile_TTD_[300:400]
batch_drug_names = TTD_validate_drugs
batch_smile_seq = TTD_validate_smiles
batch_cell_line_name = list(df_data['cell_line_name'])[0:100] + list(df_data['cell_line_name'])[0:50]
batch_drug_response = list(df_data['drug_response'])[0:100] + list(df_data['drug_response'])[0:50]
drug_atom_one_hot_chunk, drug_rel_position_chunk, edge_type_matrix_chunk,\
drug_smile_length_chunk, gene_expression_bin_chunk, gene_mutation_bin_chunk, gene_prior_chunk = \
extract_input_data_midi(batch_drug_names, batch_smile_seq, \
                        batch_cell_line_name, batch_drug_response, continuous_gene_exp, mutation_gene)


batch_shape = drug_atom_one_hot_chunk.shape[0]
mask = tf.range(start=0, limit=100, dtype=tf.float32)
mask = tf.broadcast_to(tf.expand_dims(mask,axis=0),shape=[batch_shape,100])
mask = tf.reshape(mask, shape=(batch_shape*100))
mask = mask < tf.cast(tf.repeat(drug_smile_length_chunk,repeats=100),tf.float32)
mask = tf.where(mask,1,0)
mask = tf.reshape(mask, shape=(batch_shape,100))
mask = tf.expand_dims(mask, axis=-1)



feature_select_score_model_drug = att_score_self_enco(midi_model_binary_gene_embedding,7)
feature_select_score_model_gene = att_score_self_enco(midi_model_binary_gene_embedding,31)

feature_select_score_drug = feature_select_score_model_drug.predict((drug_atom_one_hot_chunk, gene_expression_bin_chunk, \
                                                                    drug_smile_length_chunk, drug_rel_position_chunk, \
                                                                    edge_type_matrix_chunk, gene_mutation_bin_chunk, mask))[1]
feature_select_score_gene = feature_select_score_model_gene.predict((drug_atom_one_hot_chunk, gene_expression_bin_chunk, \
                                                                    drug_smile_length_chunk, drug_rel_position_chunk, \
                                                                    edge_type_matrix_chunk, gene_mutation_bin_chunk, mask))[1][:,0,:]







