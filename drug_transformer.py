import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
#import tensorflow_addons as tfa
import keras.backend as K
from tensorflow.keras import initializers
from sklearn.metrics import f1_score
from utils.smile_rel_dist_interpreter import *
#from utils.process_data import *


class masked_softmax(tf.keras.layers.Layer):
	def __init__(self, value=-1e7):
		super().__init__()
		self.value = value
		self.temperature = 9

	def call(self, X, if_sparse_max=False, valid_lens=None, **kwargs):
		"""
		Parameters:
		-----------
		X: 2D tensor specifying the attention matrix [query_seq_length, key_seq_length]
		"""
		if valid_lens == None:
			if if_sparse_max == True:
				return tfa.activations.sparsemax(X)
			else:
				X = X * self.temperature
				return tf.nn.softmax(X, axis=-1)
			#print("Im here")
		else:
			shape_X = tf.shape(X)
			X = tf.reshape(X, shape=(-1, X.shape[-1]))
			reshape_X = tf.shape(X)
			maxlen = X.shape[1]
			mask = tf.range(start=0, limit=shape_X[-1], dtype=tf.float32)[None,:]
			mask = tf.broadcast_to(mask, shape=reshape_X)

			valid_lens = tf.repeat(valid_lens, repeats = shape_X[1])
			mask = mask < tf.cast(valid_lens[:, None], dtype=tf.float32)

			X = tf.where(mask, X, self.value)

			if if_sparse_max == True:
				return tfa.activations.sparsemax(tf.reshape(X, shape=shape_X), axis=-1)
			else:
				return tf.nn.softmax(tf.reshape(X, shape=shape_X), axis=-1)


class masked_softmax_sliding_window(tf.keras.layers.Layer):
	"""
	Assign sliding window mask

	Parameters:
	-----------
	top_k: sliding window size
	step_size: sliding window step size
	"""
	def __init__(self, top_k = 200, value=-1e7, maxval=5843, step_size=30):
		super().__init__()
		self.value = value
		self.top_k = top_k
		self.maxval = maxval
		self.step_size = step_size

	def call(self, X, **kwargs):
		"""
		Parameters:
		-----------
		X: 2D tensor specifying the attention score matrix (before softmax) [query_seq_length, key_seq_length]
		"""
		shape_X = tf.shape(X)
		mask = tf.zeros((shape_X[1],shape_X[-1]), dtype=bool)

		select_mask = tf.ones((shape_X[1]*self.top_k), dtype=bool)

		gene_indices = tf.range(start=0, limit=shape_X[1], dtype=tf.dtypes.int32)
		gene_indices = tf.repeat(gene_indices, repeats = self.top_k)

		slide_single = tf.range(start=0, limit=self.top_k, dtype=tf.dtypes.int32)
		slide_window = tf.expand_dims(slide_single, axis=0)
		slide_window = tf.broadcast_to(slide_window, shape=(shape_X[1], self.top_k))

		max_value_step = self.maxval - self.top_k
		step_ = tf.cast(tf.range(start=0, limit=max_value_step, delta=self.step_size), dtype=tf.float32)
		pad_size = tf.cast(self.maxval - tf.shape(step_)[0], dtype=tf.float32)

		#pad_window = tf.constant(step_[-1],shape=(pad_size,))

		pad_window = tf.ones(shape=(pad_size,))*step_[-1]

		step_ = tf.concat([step_, pad_window], axis=-1)

		step_ = tf.expand_dims(step_, axis=1)
		step_ = tf.cast(tf.broadcast_to(step_, shape=(shape_X[1], self.top_k)),tf.int32)

		slide_window_indices = tf.math.add(slide_window, step_)

		slide_window_indices = tf.reshape(slide_window_indices, shape=(shape_X[1]*self.top_k,))


		#random_indices = tf.random.uniform(shape=[shape_X[1]*self.top_k], minval=0, maxval=self.maxval, dtype=tf.dtypes.int32)

		gene_indices = tf.stack([gene_indices, slide_window_indices],axis=-1)

		mask = tf.tensor_scatter_nd_update(mask, gene_indices, select_mask)
		mask = tf.expand_dims(mask, axis=0) 
		mask = tf.broadcast_to(mask, shape=shape_X)

		X = tf.reshape(X, shape=(-1, X.shape[-1]))
		mask = tf.reshape(mask, shape=(-1, X.shape[-1]))
		reshape_X = tf.shape(X)


		X_slide_window = tf.where(mask, X, self.value)

		mask = tf.reshape(mask, shape=shape_X)


		return tf.nn.softmax(tf.reshape(X_slide_window, shape=shape_X), axis=-1)

class masked_softmax_random(tf.keras.layers.Layer):
	"""
	Assign random mask to attentions(query->key)
	Algorithm: for each query token, select random corresponded key tokens,
	then assign mask to those tokens.

	Parameters:
	-----------
	top_k: customized variable indicating how many random tokens to be selected
	value: the negative infinity value in order to make softmax probability zero
	"""
	def __init__(self, top_k = 100, value=-1e7, maxval=5843):
		super().__init__()
		self.value = value
		self.top_k = top_k
		self.maxval = maxval

	def call(self, X, **kwargs):
		"""
		Parameters:
		-----------
		X: 2D tensor specifying the attention score matrix (before softmax) [query_seq_length, key_seq_length]
		"""
		shape_X = tf.shape(X)
		mask = tf.zeros((shape_X[1],shape_X[-1]), dtype=bool)

		select_mask = tf.ones((shape_X[1]*self.top_k), dtype=bool)

		gene_indices = tf.range(start=0, limit=shape_X[1], dtype=tf.dtypes.int32)
		gene_indices = tf.repeat(gene_indices, repeats = self.top_k)

		random_indices = tf.random.uniform(shape=[shape_X[1]*self.top_k], minval=0, maxval=self.maxval, dtype=tf.dtypes.int32)

		gene_indices = tf.stack([gene_indices, random_indices],axis=-1)

		mask = tf.tensor_scatter_nd_update(mask, gene_indices, select_mask)
		mask = tf.expand_dims(mask, axis=0)
		mask = tf.broadcast_to(mask, shape=shape_X)

		X = tf.reshape(X, shape=(-1, X.shape[-1]))
		mask = tf.reshape(mask, shape=(-1, X.shape[-1]))
		reshape_X = tf.shape(X)


		X_random = tf.where(mask, X, self.value)

		mask = tf.reshape(mask, shape=shape_X)


		return tf.nn.softmax(tf.reshape(X_random, shape=shape_X), axis=-1), 
		tf.nn.softmax(tf.reshape(X, shape=shape_X), axis=-1), mask

class masked_softmax_selected(tf.keras.layers.Layer):
	"""
	Define selected maksed for softmax layer.
	Algorithm: for the calculated dot product scores, first select
	the customized number of attention slots for picking features
	"""
	def __init__(self, top_k = 10, value=-1e7):
		super().__init__()
		self.value = value
		self.top_k = top_k

	def call(self, X, **kwargs):
		"""
		Parameters:
		-----------
		X: 2D tensor specifying the attention score matrix (before softmax) [query_seq_length, key_seq_length]
		"""
		shape_X = tf.shape(X)
		X = tf.reshape(X, shape=(-1, X.shape[-1]))
		X_top_index = tf.math.top_k(X, k=self.top_k)[0]
		X_top_index = tf.reduce_min(X_top_index, axis=-1)
		X_top_index = tf.expand_dims(X_top_index, axis=1)
		reshape_X = tf.shape(X)
		X_top_index = tf.broadcast_to(X_top_index, shape=reshape_X)

		mask = X >= X_top_index
		#maxlen = X.shape[1]
		#mask = tf.range(start=0, limit=shape_X[-1], dtype=tf.float32)[None,:]
		#mask = tf.broadcast_to(mask, shape=reshape_X)

		#valid_lens = tf.repeat(valid_lens, repeats = shape_X[1])
		#mask = mask < tf.cast(valid_lens[:, None], dtype=tf.float32)

		X = tf.where(mask, X, self.value)

		return tf.nn.softmax(tf.reshape(X, shape=shape_X), axis=-1)


class positionalencoding(tf.keras.layers.Layer):  
    """Positional encoding."""
    def __init__(self, num_hiddens, num_length, max_len=7000):
        super().__init__()
        #self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P
        self.num_length = num_length
        self.max_len = max_len
        self.num_hiddens = num_hiddens
    def call(self, X, **kwargs):
        #X = X + self.P[:, :X.shape[1], :]
        #return self.dropout(X, **kwargs)
        self.P = np.zeros((1, self.max_len, self.num_hiddens))
        XX = np.arange(self.max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
                0, self.num_hiddens, 2, dtype=np.float32) / self.num_hiddens)
        self.P[:, :, 0::2] = np.sin(XX)
        self.P[:, :, 1::2] = np.cos(XX)
        shape_X = tf.shape(X)
        X = tf.math.l2_normalize(X, axis=-1)
        self.P = tf.cast(tf.math.l2_normalize(self.P[:, :self.num_length,:], axis=-1), 
            dtype=tf.float32)

        return tf.cast(tf.math.l2_normalize(tf.math.add(X, self.P), axis=-1), dtype=tf.float32)


class position_wise_embedding(tf.keras.layers.Layer):
	"""
	Define position wise embedding for sequence input embedding

	Parameters:
	-----------
	kernel: the embedding matrix
	"""
	def __init__(self, output_dim):
		super().__init__()
		self.output_dim = output_dim

	def build(self, input_shape, **kwargs):
		self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], self.output_dim),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(
			initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

	def call(self, input_data, **kwargs):
		output_embedding = tf.keras.activations.relu(tf.matmul(input_data, self.kernel) + self.b)

		return tf.cast(tf.math.l2_normalize(output_embedding, axis=-1), dtype=tf.float32)

class feature_selection_layer(tf.keras.layers.Layer):
	"""
	Define feature selection layer using greedy trainable 
	feature selection:
	https://openreview.net/pdf?id=TTLLGx3eet
	the output dimension is one, and input through a softmax layer 
	for representing the selection probability.

	Parameters:
	-----------
	select_dim: the feature dimensions to be selected.

	Returns:
	--------
	select_indices: the selected feature indices.
	"""
	def __init__(self, select_dim=200):
		super().__init__()
		self.select_dim = select_dim
		#self.output_dim = 1

	def build(self, input_shape, **kwargs):
		self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], 1),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)
		#b_init = tf.zeros_initializer()
		#self.b = tf.Variable(
		#	initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

	def call(self, input_data, **kwargs):
		output_score = tf.matmul(input_data, self.kernel)
		shape_score = tf.shape(output_score)
		#print(output_score.shape)
		#output_score = tf.reduce_mean(output_score, axis=0)
		output_score = tf.nn.softmax(output_score,axis=1)
		#output_score_ = tf.reshape(output_score,shape=[shape_score[0],shape_score[1]])
		#print(output_score.shape)
		#top_indices = tf.math.top_k(output_score_, k=self.select_dim).indices
		#print(top_indices.shape)
		#output_embedding = tf.gather(input_data, indices=top_indices, batch_dims=1)
		#output_score_ = tf.gather(output_score, indices=top_indices, batch_dims=1)
		#output_score_ = tf.nn.softmax(output_score_,axis=1)
		#shape_score_ = tf.shape(output_score_)
		#output_score_ = tf.reshape(output_score_, shape = [shape_score_[0],shape_score_[1]])
		#output_embedding = tf.math.multiply(output_embedding, output_score_)
		#output_embedding = tf.cast(tf.math.l2_normalize(tf.math.reduce_sum(output_embedding, axis=1)), dtype=tf.float32)

		#return tf.cast(top_indices, tf.int32), output_score, output_embedding
		return output_score

class feature_selection_layer_global_drug(tf.keras.layers.Layer):
	"""
	Define feature selection layer using greedy trainable 
	feature selection:
	https://openreview.net/pdf?id=TTLLGx3eet
	the output dimension is one, and input through a softmax layer 
	for representing the selection probability.

	Parameters:
	-----------
	select_dim: the feature dimensions to be selected.

	Returns:
	--------
	select_indices: the selected feature indices.
	"""
	def __init__(self, select_dim=200):
		super().__init__()
		self.select_dim = select_dim
		#self.output_dim = 1

	def build(self, input_shape, **kwargs):
		self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], 5843),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)
		#b_init = tf.zeros_initializer()
		#self.b = tf.Variable(
		#	initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

	def call(self, input_data, **kwargs):
		output_score = tf.matmul(input_data, self.kernel)
		shape_score = tf.shape(output_score)
		#print(output_score.shape)
		#output_score = tf.reduce_mean(output_score, axis=0)
		output_score = tf.nn.softmax(output_score,axis=-1)
		return output_score


class dotproductattention(tf.keras.layers.Layer):  #@save
	"""
	Define scaled dot product layer

	Parameters:
	-----------
	kernel_key: embedding matrix for key
	kernel_value: embedding matrix for value
	kernel_query: embedding matrix for query

	Returns:
	--------
	attention_score: the scale dot product score
	"""
	def __init__(self, output_dim, if_select_feature=None, if_self_att=False):
	    super().__init__()
	    self.output_dim = output_dim
	    self.if_select_feature = if_select_feature
	    self.if_self_att = if_self_att
	    #self.relative_encoding_lookup = relative_encoding_lookup
	    #self.masked_softmax = masked_softmax()

	    #self.kernel_key = tf.keras.layers.Dense(output_dim, activation='sigmoid', 
	    #	kernel_regularizer=regularizers.L2(1e-4))

	    #self.kernel_query = tf.keras.layers.Dense(output_dim, activation='sigmoid', 
	    #	kernel_regularizer=regularizers.L2(1e-4))

	    #self.kernel_value = tf.keras.layers.Dense(output_dim, kernel_initializer=initializers.RandomNormal(seed=42),
	                                         #kernel_regularizer=regularizers.L2(1e-4),
	                                         #bias_initializer=initializers.Zeros())



	def build(self, input_shape):
		b_init = tf.zeros_initializer()

		if self.if_self_att == False:
			self.kernel_key = self.add_weight(name='kernel_key', shape = (input_shape[-1], self.output_dim),
				initializer = tf.keras.initializers.RandomNormal(seed=42), trainable = True)

			self.b_key = tf.Variable(name='bias_key', 
				initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

		#self.b_key = self.add_weight(name='bias_key',shape = (self.output_dim,),
		#initializer = tf.keras.initializers.RandomNormal(seed=42), trainable = True)

		self.kernel_query  = self.add_weight(name='kernel_query', shape = (input_shape[-1], self.output_dim),
			initializer = tf.keras.initializers.RandomNormal(seed=42), trainable = True)

		self.b_query = tf.Variable(name='bias_query', 
			initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

		#self.b_query = self.add_weight(name='bias_query',shape = (self.output_dim,),
		#initializer = tf.keras.initializers.RandomNormal(seed=42), trainable = True)

		if self.if_select_feature == None:
			self.kernel_value = self.add_weight(name='kernel_value', shape=(input_shape[-1], self.output_dim),
				initializer=tf.keras.initializers.he_normal(seed=42), trainable=True)

			self.b_value = tf.Variable(name='bias_value',
				initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

	    #self.b_value = self.add_weight(name='bias_value',shape = (self.output_dim,),
            #initializer = tf.keras.initializers.RandomNormal(seed=42), trainable = True)


	def call(self, queries, keys, values, relative_encoding_lookup=None, edge_type_embedding=None, if_select_feature=None,**kwargs):
		d = queries.shape[-1]
		queries = tf.math.l2_normalize(tf.matmul(queries, self.kernel_query) + self.b_query, axis=-1)
		#queries = tf.matmul(queries, self.kernel_query) + self.b_query
		shape = tf.shape(queries)
		#queries = self.kernel_query(queries)
		"""
		change the kernel_key to kernel_query in order for self-att matrix to be consistent
		"""
		if self.if_self_att == False:
			keys = tf.math.l2_normalize(tf.matmul(keys, self.kernel_key) + self.b_key, axis=-1)
		else:
			keys = tf.math.l2_normalize(tf.matmul(keys, self.kernel_query) + self.b_query, axis=-1)
		#keys = tf.matmul(keys, self.kernel_key) + self.b_key
		#keys = self.kernel_key(keys)
		if if_select_feature == None:
			values = tf.math.l2_normalize(tf.matmul(values, self.kernel_value) + self.b_value, axis=-1)
		#values = tf.matmul(values, self.kernel_value) + self.b_value
		#values = self.kernel_value(values)


		if relative_encoding_lookup == None:
			scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(tf.cast(d, dtype=tf.float32))

			if if_select_feature == None:
				return scores, values, queries, keys
			else:
				return scores, queries, keys
		else:
			scores_ = tf.matmul(queries, keys, transpose_b=True)*0.3
			#print("scores_ shape")
			#print(scores_.shape)
			#queries_origin_ = tf.concat(queries_origin_, relative_encoding_origin)
			#queries_origin = tf.expand_dims(queries, axis=1)
			#queries_origin = tf.broadcast_to(queries_origin, [shape[0],shape[1],shape[1],shape[-1]])
			#queries_origin = tf.math.add(queries_origin, relative_encoding_origin)
			#queries_origin = tf.concat((queries_origin, relative_encoding_origin),axis=-1)
			queries_ = tf.expand_dims(queries, axis=2)
			queries_ = tf.broadcast_to(queries_, [shape[0],shape[1],shape[1],shape[-1]])
			#queries_ = tf.concat((queries_, relative_encoding_lookup),axis=-1)
			#queries_ = tf.math.add(queries_, relative_encoding_lookup)
			#print(queries_.shape)
			#relative_encoding_lookup = tf.expand_dims(relative_encoding_lookup,axis=0)
			#relative_encoding_lookup = tf.broadcast_to(relative_encoding_lookup,[shape[0],shape[1],shape[1],shape[-1]])
			scores_position = tf.reduce_sum(tf.multiply(queries_, relative_encoding_lookup), axis=-1)*0.5
			scores_edge_embedding = tf.reduce_sum(tf.multiply(queries_, edge_type_embedding), axis=-1)*0.3
			#print(scores_position.shape)
			#scores = tf.reduce_sum(tf.multiply(queries_origin, queries_), axis=-1)
			scores_ = tf.add(scores_, scores_position)
			scores = tf.add(scores_, scores_edge_embedding)
			scores = scores/tf.math.sqrt(tf.cast(d, dtype=tf.float32))
			#print(scores.shape)


			#self.attention_weights = self.masked_softmax(scores, valid_lens)
			if if_select_feature == None:	
				return scores, values, queries, keys
			else:
				return scores, queries, keys

class attention_embedding(tf.keras.layers.Layer):
	"""
	Define attention embedding layer, perform dummy
	attention multiplication with value input

	Parameters:
	-----------
	att_weights: the attention score matrix
	input_value: input value matrix

	Returns:
	--------

	the attention embedding matrix
	"""
	def __init__(self, output_dim):
		super().__init__()
		self.output_dim = output_dim

		self.kernel_position = tf.keras.layers.Dense(output_dim, kernel_initializer=initializers.RandomNormal(seed=42),
			activation='relu', 
			kernel_regularizer=regularizers.L2(1e-4))

	def call(self, att_weights, input_value,relative_encoding_lookup=None, edge_type_embedding=None,**kwargs):

		if relative_encoding_lookup == None:
			return tf.cast(tf.matmul(att_weights, input_value), dtype=tf.float32)
		else:
			shape = tf.shape(input_value)
			value_ = tf.expand_dims(input_value, axis=1)
			value_ = tf.broadcast_to(value_, [shape[0],shape[1],shape[1],shape[-1]])
			#relative_encoding_lookup_ = self.kernel_position(relative_encoding_lookup)
			value_ = value_ 
			relative_encoding_lookup = relative_encoding_lookup 
			edge_type_embedding = edge_type_embedding 
			value_ = tf.math.add(value_, relative_encoding_lookup)
			value_ = tf.math.add(value_, edge_type_embedding)
			#value_ = tf.concat((value_, relative_encoding_lookup),axis=-1)
			#value_ = self.kernel_position(value_)
			att_weights_ = tf.expand_dims(att_weights, axis=-1)
			att_weights_ = tf.broadcast_to(att_weights_, [shape[0],shape[1],shape[1],shape[-1]])

			return tf.cast(tf.reduce_sum(tf.multiply(att_weights_, value_), 
				axis=-2), dtype=tf.float32)
		#return tf.cast(tf.matmul(att_weights, input_value), dtype=tf.float32)


class residual_connection(tf.keras.layers.Layer):
    """
    Define residual connection layer
    """
    def __init__(self):  
        super().__init__()

    def call(self, X, Y, **kwargs):
        #X = tf.math.l2_normalize(X, axis=-1)
        #Y = tf.math.l2_normalize(Y, axis=-1)
        return tf.cast(tf.math.l2_normalize(tf.math.add(X,Y), axis=-1), dtype=tf.float32)
        #return tf.cast(tf.math.add(X,Y), dtype=tf.float32)


class feed_forward_layer(tf.keras.layers.Layer):
	"""
	Define a dummy feed forward layer for intigrating 
	gene expression data

	Parameters:
	-----------
	kernel: the trainable kernel matrix

	Returns:
	--------
	the embedding vector
	"""
	def __init__(self, output_dim):
		super().__init__()
		self.output_dim = output_dim
		self.dropout = tf.keras.layers.Dropout(0.1)

	def build(self, input_shape, **kwargs):
		self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], self.output_dim),
		initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(
			initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

	def call(self, input_data, **kwargs):
		output_embedding = tf.keras.activations.relu(tf.matmul(input_data, self.kernel) + self.b)

		#return self.dropout(tf.cast(tf.math.l2_normalize(output_embedding, axis=-1), dtype=tf.float32),**kwargs)
		return tf.cast(tf.math.l2_normalize(output_embedding, axis=-1), dtype=tf.float32)


class concatenation_layer(tf.keras.layers.Layer):
	"""
	Define concatenation layer
	"""
	def __init__(self):
		super().__init__()

	def call(self, X, Y, **kwargs):
		X = tf.math.l2_normalize(X, axis=-1)
		Y = tf.math.l2_normalize(Y, axis=-1)

		return tf.cast(tf.math.l2_normalize(tf.concat([X,Y],axis=1), axis=-1), dtype=tf.float32)


class encoder_block(tf.keras.layers.Layer):
	"""
	Define self attention encoder block, add the position encoding
	since drug smile sequence has the order information

	Parameters:
	-----------
	num_hiddens: the hidden dimension for embedding matrix
	seq_length: the length for input sequence

	Returns:
	-------
	encoder_embedding: the encoder embedding output
	att_score: the self attention score
	"""
	def __init__(self, num_hiddens, seq_length, if_self_att_=True):
		super().__init__()
		self.masked_softmax = masked_softmax()
		self.pos_encoding = positionalencoding(num_hiddens,seq_length)
		self.dotproductattention = dotproductattention(num_hiddens, if_self_att=if_self_att_)
		self.att_embedding = attention_embedding(num_hiddens)
		self.r_connection = residual_connection()

	def call(self, X, if_sparse_max=False, enc_valid_lens=None, relative_pos_enc=None, edge_type_enc=None, **kwargs):
		#X = self.pos_encoding(X)
		score, value, query, keys = self.dotproductattention(X,X,X,relative_encoding_lookup=relative_pos_enc, edge_type_embedding=edge_type_enc)
		value = tf.math.l2_normalize(value, axis=-1)
		att_score = self.masked_softmax(score, if_sparse_max, enc_valid_lens)
		#print(att_score.shape)
		att_embedding_ = self.att_embedding(att_score, value, relative_encoding_lookup=relative_pos_enc, edge_type_embedding=edge_type_enc)

		encoder_embedding = self.r_connection(value, att_embedding_)
		#encoder_embedding = value

		return encoder_embedding, att_score, score

class decoder_cross_block(tf.keras.layers.Layer):
	"""
	Define decoder cross attention 
	"""
	def __init__(self, num_hiddens, if_select_feature_=None, if_self_att_=False):
		super().__init__()
		self.masked_softmax_deco_cross = masked_softmax()
		self.dotproductattention_deco_cross = dotproductattention(num_hiddens, if_select_feature=if_select_feature_, if_self_att=if_self_att_)
		self.att_embedding = attention_embedding(num_hiddens)
		self.r_connection = residual_connection()

	def call(self, Y, X, if_sparse_max=False, enc_valid_lens=None, if_select_feature_=None, **kwargs):
		if if_select_feature_ == None:
			score_deco_cross, value_deco_cross, query_deco_cross,key_deco_cross = self.dotproductattention_deco_cross(Y,X,X,if_select_feature=if_select_feature_)
		else:
			score_deco_cross, query_deco_cross, key_deco_cross = self.dotproductattention_deco_cross(Y,X,X,if_select_feature=if_select_feature_)
		att_score_deco_cross = self.masked_softmax_deco_cross(score_deco_cross, if_sparse_max, enc_valid_lens)	

		if if_select_feature_ == None:
			att_embedding_deco_cross = self.att_embedding(att_score_deco_cross, value_deco_cross)


	#att_embedding_deco_cross = tf.concat([att_embedding_deco_cross, att_embedding_deco_cross2],axis=-1)
	#query_deco_cross = tf.concat([query_deco_cross, query_deco_cross2],axis=-1)
		if if_select_feature_ == None:
			cross_embedding = self.r_connection(query_deco_cross, att_embedding_deco_cross)

		if if_select_feature_ == None:
			return cross_embedding, att_score_deco_cross, value_deco_cross, score_deco_cross
		else:
			return query_deco_cross, att_score_deco_cross, key_deco_cross, score_deco_cross


class feature_selection_cross_block(tf.keras.layers.Layer):
	"""
	Define global feature selection cross attention block
	"""
	def __init__(self, num_hiddens, if_select_feature_=None, if_self_att_=False):
		super().__init__()
		self.masked_softmax_deco_cross = masked_softmax()
		self.dotproductattention_deco_cross = dotproductattention(num_hiddens, if_select_feature=if_select_feature_, if_self_att=if_self_att_)
		self.att_embedding = attention_embedding(num_hiddens)
		self.r_connection = residual_connection()

	def call(self, Y, X, gene_embedding, mutation_embedding, if_sparse_max=False, enc_valid_lens=None, if_select_feature_=None, **kwargs):
		if if_select_feature_ == None:
			score_deco_cross, value_deco_cross, query_deco_cross, key_deco_cross = self.dotproductattention_deco_cross(Y,X,X,
				gene_embedding = gene_embedding, mutation_embedding=mutation_embedding, if_select_feature=if_select_feature_)
		else:
			score_deco_cross, query_deco_cross, key_deco_cross = self.dotproductattention_deco_cross(Y,X,X, 
				gene_embedding = gene_embedding, mutation_embedding=mutation_embedding, if_select_feature=if_select_feature_)
		att_score_deco_cross = self.masked_softmax_deco_cross(score_deco_cross, if_sparse_max, enc_valid_lens)	


		att_embedding_deco_cross_gene = tf.math.add(key_deco_cross, gene_embedding)

		att_embedding_deco_cross_gene = tf.math.add(att_embedding_deco_cross_gene, mutation_embedding)

		return query_deco_cross, att_score_deco_cross, att_embedding_deco_cross_gene

class drug_transformer_():
	"""
	Implement the drug transformer model architecture
	"""
	def __init__(self, gene_embeddings):#, relative_pos_enc_lookup=None):

		#self.string_lookup = tf.keras.layers.StringLookup(vocabulary=gene_expression_vocab)
		#self.layer_one_hot = tf.keras.layers.CategoryEncoding(num_tokens=5843, output_mode="one_hot")

		#self.input_gene_expression_names = tf.constant(gene_expression_vocab)
		#self.input_gene_expression_index = self.string_lookup(self.input_gene_expression_names)-1

		#self.relative_pos_enc_lookup = relative_pos_enc_lookup

		#self.input_gene_expression_one_hot = self.layer_one_hot(self.input_gene_expression_index)

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
		"""
		mask = tf.range(start=0, limit=70, dtype=tf.float32)
		mask = tf.broadcast_to(tf.expand_dims(mask,axis=0),shape=[shape_input[0],70])
		mask = tf.reshape(mask, shape=(shape_input[0]*70,1))
		mask = mask < tf.cast(tf.repeat(enc_valid_lens_,repeats=70),tf.float32)
		mask = tf.where(mask,1,0)
		mask = tf.reshape(mask, shape=(shape_input[0],70))

		shape_x = tf.shape(X)
		mask = tf.expand_dims(mask, axis=-1)
		mask = tf.cast(tf.broadcast_to(mask, shape=shape_x),tf.float32)

		X = tf.multiply(X, mask)
		"""
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

		#Y1, att_score_deco_cross1, Y1_key = self.decoder_cross_1(Y, X, enc_valid_lens=enc_valid_lens_, if_sparse_max=False)
		#Y2, att_score_deco_cross2, Y2_key = self.decoder_cross_2(Y, X, enc_valid_lens=enc_valid_lens_, if_sparse_max=False)

		#Y = tf.concat([Y1,Y2],axis=-1)

		#Y = tf.math.add(Y1,Y2)

		#Y = self.r_connection_multi_deco_gene(Y1,Y2)

		#Y = self.dense_15(Y)

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
		#Y = self.dense_16(Y)
		#Y_global = tf.reduce_sum(Y_global, axis=1)

		#Y = self.dense_16(Y)
		#Y_global = self.r_connection_feature(Y_key, Y_global)
		#Y_global2 = tf.math.multiply(att_score_global2, Y_key2)
		#Y_global3 = tf.math.multiply(att_score_global3, Y_key3)
		#Y = tf.concat([Y_global1, Y_global2, Y_global3],axis=-1)
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


def return_drug_gene(CCLE_name_test, drug_name_list_test, gene_expression_test, drug_one_hot_encoding_test, drug_smile_length_test, ic50_list_test, drug_smile_list_test, drug_name):
    lung_index = []
    for i in range(len(CCLE_name_test)):
        #print(np.array(CCLE_name_test)[i][-4:])
        #if np.array(CCLE_name_test)[i][-4:] == "LUNG":
        if drug_name_list_test[i] == drug_name:
            lung_index.append(i)
    drug_name_lung = [drug_name_list_test[i] for i in lung_index]
    CCLE_name_test_lung = [CCLE_name_test[i] for i in lung_index]
    drug_one_hot_encoding_test_lung = [drug_one_hot_encoding_test[i] for i in lung_index]
    gene_expression_test_lung = [gene_expression_test[i] for i in lung_index]
    drug_smile_length_test_lung = [drug_smile_length_test[i] for i in lung_index]
    ic50_list_test_lung = [ic50_list_test[i] for i in lung_index] 
    drug_smile_list_test_lung = [drug_smile_list_test[i] for i in lung_index]
    
    return np.array(drug_one_hot_encoding_test_lung),np.array(gene_expression_test_lung), \
np.array(drug_smile_length_test_lung), drug_name_lung, CCLE_name_test_lung, ic50_list_test_lung, drug_smile_list_test_lung
    

def return_select_gene_feature(gene_expression, drug_one_hot_encoding, top_gene_index=None, if_select=True):
    if not top_gene_index == None:
        gene_expression_select = tf.gather(gene_expression[:,:,0], top_gene_index, axis=1)
    shape_drug = drug_one_hot_encoding.shape
    drug_one_hot_encoding = tf.reshape(drug_one_hot_encoding, shape=[shape_drug[0],shape_drug[1]*shape_drug[2]])
    if if_select:
        cell_line_drug_feature = np.concatenate((gene_expression_select,drug_one_hot_encoding),1)
    else:
        cell_line_drug_feature = np.concatenate((gene_expression[:,:,0],drug_one_hot_encoding),1)

    return cell_line_drug_feature

def return_gene_smile_target(model, drug_one_hot, gene_expression, drug_length, batch_size=30):
    """
    return the drug's chemical structure & gene-targeting relation
    """
    feature_select_score_model1 = att_score_self_enco(model, 14)
    feature_select_score_model2 = att_score_self_enco(model, 15)
    feature_select_score_whole_model_1 = []
    feature_select_score_whole_model_2 = []
    len_train = len(drug_one_hot)
    fraction_ = int(np.floor(len_train/batch_size))
    median_train = int(np.ceil(len_train/2))
    for i in range(fraction_):
        feature_select_score1 = feature_select_score_model1.predict((drug_one_hot[i*batch_size:(i+1)*batch_size], 
                                                                     gene_expression[i*batch_size:(i+1)*batch_size], np.array(drug_length[i*batch_size:(i+1)*batch_size])))[1]
        feature_select_score2 = feature_select_score_model2.predict((drug_one_hot[i*batch_size:(i+1)*batch_size], 
                                                                     gene_expression[i*batch_size:(i+1)*batch_size], np.array(drug_length[i*batch_size:(i+1)*batch_size])))[1]
        feature_select_score_whole_model_1.append(feature_select_score1)
        feature_select_score_whole_model_2.append(feature_select_score2)

    feature_select_score_whole_model_1 = tf.concat(feature_select_score_whole_model_1,axis=0)
    feature_select_score_whole_model_2 = tf.concat(feature_select_score_whole_model_2,axis=0)

    feature_select_score_whole_model_1 = tf.math.reduce_mean(feature_select_score_whole_model_1, axis=0)
    feature_select_score_whole_model_2 = tf.math.reduce_mean(feature_select_score_whole_model_2, axis=0)

    return feature_select_score_whole_model_1, feature_select_score_whole_model_2   

def return_gene_drug_target_train(model, gene_names, drug_lung, gene_lung, drug_lung_length,CCLE_name_lung, drug_name_lung,drug_smile_lung, ic50_lung,top_gene=5370, batch_size=10):
    """
    return the gene-drug targeting cross-attention matrix
    """
    feature_select_score_model1 = att_score_self_enco(model,27)
    feature_select_score_model2 = att_score_self_enco(model,28)
    feature_select_score_model3 = att_score_self_enco(model,29)
    feature_select_score_whole = []
    len_train = len(drug_lung)
    #print(len_train)
    #print("this is wrong")
    faction_ = int(np.floor(len_train/batch_size))
    median_train = int(np.ceil(len_train/2))
    for i in range(faction_):
        feature_select_score1 = feature_select_score_model1.predict((drug_lung[i*batch_size:(i+1)*batch_size], 
                                                                     gene_lung[i*batch_size:(i+1)*batch_size], np.array(drug_lung_length[i*batch_size:(i+1)*batch_size])))
        feature_select_score2 = feature_select_score_model2.predict((drug_lung[i*batch_size:(i+1)*batch_size], 
                                                                     gene_lung[i*batch_size:(i+1)*batch_size], np.array(drug_lung_length[i*batch_size:(i+1)*batch_size])))
        feature_select_score3 = feature_select_score_model3.predict((drug_lung[i*batch_size:(i+1)*batch_size], 
                                                                     gene_lung[i*batch_size:(i+1)*batch_size], np.array(drug_lung_length[i*batch_size:(i+1)*batch_size])))
        
        #print(feature_select_score1[1].shape)
        #print(feature_select_score2[1].shape)
        #print(feature_select_score3[1].shape)
        feature_select_score = tf.concat([feature_select_score1[1], feature_select_score2[1], feature_select_score3[1]],axis=1)
        
        feature_select_score = tf.sort(feature_select_score, axis=1)
        #feature_select_score_ = feature_select_score[:,1,:]
        feature_select_score_ = tf.math.reduce_mean(feature_select_score, axis=1)
        
        feature_select_score_whole.append(feature_select_score_)

    print(feature_select_score_whole)
    feature_select_score_whole_ = tf.concat(feature_select_score_whole,axis=0)
    #print(feature_select_score_whole_)
    #feature_select_score_optimal = feature_select_score_whole_[median_train,:]
    feature_select_score_optimal = tf.math.reduce_mean(feature_select_score_whole_,axis=0)
    #print(feature_select_score_optimal.shape)
    
    test_length = len(CCLE_name_lung)
    top_genes_score, top_genes_index = tf.math.top_k(feature_select_score_optimal, k=top_gene)
    top_gene_names = np.array([gene_names[i] for i in top_genes_index])

    return top_gene_names, top_genes_score, top_genes_index

def generate_chunk_data(model_midi, P, gene_expression_update, drug_smile_list_update,ic50_list_update, 
                        mutation_gene_update,string_lookup, layer_one_hot,
                        training_chunk_size,smile_length=70):
    
    drug_rel_position_chunk = []
    drug_smile_length_update = []
    edge_type_matrix_chunk = []
    input_drug_atom_one_hot_chunk = []
    gene_expression_update_chunk = []
    gene_mutation_update_chunk = []
    ic50_list_update_chunk = []
    drug_rel_position_origin_chunk = []
    for j in range(training_chunk_size):
        #print(i)
        gene_expression_update_chunk.append(gene_expression_update[j,:,:])
        rel_distance_ = generate_rel_dist_matrix(drug_smile_list_update[j])
        interpret_smile_ = generate_interpret_smile(drug_smile_list_update[j])
        ic50_list_update_chunk.append(ic50_list_update[j])
        #rel_distance_train.append(rel_distance_)
        #interpret_smile_train.append(interpret_smile_[0])
        gene_mutation_update_chunk.append(mutation_gene_update[j])
        #projection_train.append(projection_)
    
        shape = rel_distance_.shape[0]
        #shape_ = rel_distance_.shape
        #drug_rel_position = tf.gather(P[0], tf.cast(rel_distance_+130,tf.int32), axis=0)
        #rel_distance_origin = np.zeros(shape=shape_)
        #drug_rel_position_origin = tf.cast(tf.gather(P[0], tf.cast(rel_distance_origin,tf.int32), axis=0), tf.float32)
        drug_rel_position = tf.cast(tf.gather(P[0], tf.cast(rel_distance_,tf.int32), axis=0), tf.float32)
        concat_left = tf.cast(tf.zeros((smile_length-shape,shape,60)), tf.float32)
        concat_right = tf.cast(tf.zeros((smile_length,smile_length-shape,60)), tf.float32)
        drug_rel_position = tf.concat((drug_rel_position,concat_left),axis=0)
        drug_rel_position = tf.concat((drug_rel_position,concat_right),axis=1)
        drug_rel_position_chunk.append(drug_rel_position)
    
        edge_type_matrix = get_drug_edge_type(drug_smile_list_update[j])
        shape = edge_type_matrix.shape[0]
        edge_type_matrix = tf.gather(edge_type_dict,tf.cast(edge_type_matrix,tf.int16),axis=0)
        #drug_rel_position = tf.cast(tf.gather(P[0], tf.cast(rel_distance_,tf.int32), axis=0), tf.float32)
        concat_left = tf.zeros((smile_length-shape,shape,5))
        concat_right = tf.zeros((smile_length,smile_length-shape,5))
        edge_type_matrix = tf.concat((edge_type_matrix,concat_left),axis=0)
        edge_type_matrix = tf.concat((edge_type_matrix,concat_right),axis=1)
        edge_type_matrix_chunk.append(edge_type_matrix)
    
        #drug_rel_position_origin = tf.concat((drug_rel_position_origin,concat_left),axis=0)
        #drug_rel_position_origin = tf.concat((drug_rel_position_origin,concat_right),axis=1)
        #drug_rel_position_origin_chunk.append(drug_rel_position_origin)
    
        """
        getting drug one hot embeddings
        """
        input_drug_atom_names = tf.constant(list(interpret_smile_[0]))
        input_drug_atom_index = string_lookup(input_drug_atom_names)-1
        input_drug_atom_one_hot = layer_one_hot(input_drug_atom_index)
        shape_drug_miss = input_drug_atom_one_hot.shape[0]
        concat_right = tf.zeros((smile_length-shape_drug_miss,8))
        input_drug_atom_one_hot = tf.concat((input_drug_atom_one_hot,concat_right),axis=0)
        drug_smile_length_update.append(shape_drug_miss)
        input_drug_atom_one_hot_chunk.append(input_drug_atom_one_hot)
    
    drug_rel_position_chunk = tf.stack(drug_rel_position_chunk)
    edge_type_matrix_chunk = tf.stack(edge_type_matrix_chunk)
    #drug_rel_position_origin_chunk = tf.stack(drug_rel_position_origin_chunk)
    input_drug_atom_one_hot_chunk = tf.stack(input_drug_atom_one_hot_chunk)
    drug_smile_length_update = np.array(drug_smile_length_update)
    gene_expression_update_chunk = tf.stack(gene_expression_update_chunk)
    gene_mutation_update_chunk = tf.stack(gene_mutation_update_chunk)
    
    return input_drug_atom_one_hot_chunk, gene_expression_update_chunk, drug_smile_length_update,drug_rel_position_chunk,\
    edge_type_matrix_chunk,gene_mutation_update_chunk,np.array(ic50_list_update_chunk)















