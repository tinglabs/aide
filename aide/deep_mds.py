"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import tensorflow as tf

from aide.aide_ import AIDE, AIDEModel
from aide.model_config import Config
from aide.utils_tf import get_optimizer, get_dist_func
from aide.constant import OPTIMIZER_ADAM, RELU, TRAIN_MODE, EVAL_MODE, PREDICT_MODE
from aide.constant import MDS_LOSS_ABS_STRESS


class DeepMDSConfig(Config):
	def __init__(self, path=None, assign_dict=None):
		super(DeepMDSConfig, self).__init__()
		self.optimizer = OPTIMIZER_ADAM
		self.lr = 0.0001  # Learning rate
		self.optimizer_kwargs = {}

		self.alpha = 1.0                        # Weight of MDS loss: L = reconstruct_loss + mds_loss * alpha + l2_loss * w_decay
		self.w_decay = 0.0                      # Weight of l2 norm loss

		self.mds_units = [1024, 512, 256]       # Units of MDS Encoder: n_features*1024 - relu - 1024*512 - relu - 512*256 - none
		self.mds_acts = [RELU, RELU, None]

		self.dist_name = 'euclidean'            # 'euclidean' | 'manhattan' | 'chebyshev' | 'cosine' | 'pearson'
		self.mds_loss = MDS_LOSS_ABS_STRESS     # MDS_LOSS_ABS_STRESS | MDS_LOSS_S_STRESS | MDS_LOSS_RAW_STRESS | MDS_LOSS_NORM_STRESS | MDS_LOSS_SQUARE_STRESS_1 | MDS_LOSS_ELASTIC | MDS_LOSS_SAMMON
		self.dist_eps = 1e-6                    # Avoid 'nan' during back propagation

		self.max_step_num = 20000               # Maximize Number of batches to run
		self.min_step_num = 4000                # Minimize number of batches to run
		self.early_stop_patience = 6            # None | int: Training will stop when no improvement is shown during (early_stop_patience * val_freq) epoches. Set to None if early stopping is not used.

		self.print_freq = 50                    # Print train loss every print_freq epoches
		self.val_freq = 100                   # Calculate validation loss every val_freq epoches (Note that it is used for early stopping)
		self.draw_freq = 500                    # Draw
		self.save_model = False                 # Whether to save model
		self.verbose = True

		self.batch_size = 256  # (batch_size * 2) samples will be fed in each batch during training
		self.validate_size = 2560  # validate_size samples will be used as validation set
		self.embed_batch_size = 2560  # embed_batch_size samples will be fed in each batch during generating embeddings

		# Will be set automatically
		self.n_samples = None
		self.n_features = None
		self.issparse = None
		self.dtype = None
		self.feed_type = None
		self.train_tfrecord_path = None
		self.pred_tfrecord_path = None

		if path is not None:
			self.load(path)

		if assign_dict is not None:
			self.assign(assign_dict)


class DeepMDSModel(AIDEModel):
	def __init__(self, mode, config, batch):
		super(DeepMDSModel, self).__init__(mode, config, batch)


	def forward(self, batch):
		X = batch
		c = self.config
		self.cal_dist = get_dist_func(c.dist_name, sparse=False)

		if type(X) == tf.SparseTensor:
			X = tf.sparse_tensor_to_dense(X, validate_indices=False)

		if X.get_shape().as_list()[-1] is None:
			X = tf.reshape(X, (-1, c.n_features))

		with tf.variable_scope('MDS'):
			self.mds_h = self.encoder(X, c.mds_units, c.mds_acts, 0.0)

		if self.mode == PREDICT_MODE:
			return

		# loss
		self.reconstruct_loss = self.mds_loss = self.l2_loss = tf.constant(0., dtype=tf.float32)

		pair_num = tf.cast(tf.shape(self.mds_h)[0] / 2, tf.int32)
		h1, h2 = self.mds_h[:pair_num], self.mds_h[pair_num:]
		dist = self.cal_dist(X[:pair_num], X[pair_num:])
		self.mds_loss = self.get_mds_loss(c.mds_loss, h1, h2, dist, c.dist_eps)

		if c.w_decay > 1e-8:   # l2 loss
			self.l2_loss = self.get_l2_loss(c.w_decay)

		self.loss = self.mds_loss * c.alpha + self.l2_loss
		self.all_loss = [self.reconstruct_loss, self.mds_loss, self.l2_loss]

		if self.mode == EVAL_MODE:
			return

		# optimize
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		optimizer = get_optimizer(c.optimizer, c.lr, **c.optimizer_kwargs)
		self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		self.init_op = tf.global_variables_initializer()


class DeepMDS(AIDE):
	def __init__(self, name=None, save_folder=None):
		super(DeepMDS, self).__init__(name or 'DeepMDS', save_folder)


	def pretrain(self, sess, logger, c):
		pass


	def build(self, config):
		with tf.name_scope(TRAIN_MODE):
			with tf.variable_scope('Model'):
				self.train_data, self.train_data_init_op, self.train_data_ph = self.get_train_data(config)
				self.train_model = DeepMDSModel(TRAIN_MODE, config, self.train_data)
		with tf.name_scope(EVAL_MODE):
			with tf.variable_scope('Model', reuse=True):
				self.eval_data, self.eval_data_init_op, self.eval_data_ph = self.get_eval_data(config)
				self.eval_model = DeepMDSModel(EVAL_MODE, config, self.eval_data)
		with tf.name_scope(PREDICT_MODE):
			with tf.variable_scope('Model', reuse=True):
				self.pred_data, self.pred_data_init_op, self.pred_data_ph = self.get_predict_data(config)
				self.pred_model = DeepMDSModel(PREDICT_MODE, config, self.pred_data)

