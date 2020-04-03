"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import tensorflow as tf
import numpy as np

from aide.aide_ import AIDE, AIDEModel
from aide.model_config import Config
from aide.utils_tf import get_optimizer, get_dist_func
from aide.constant import OPTIMIZER_ADAM, RELU, TRAIN_MODE, EVAL_MODE, PREDICT_MODE
from aide.constant import MDS_LOSS_ABS_STRESS


class AEConfig(Config):
	def __init__(self, path=None, assign_dict=None):
		super(AEConfig, self).__init__()
		self.optimizer = OPTIMIZER_ADAM
		self.lr = 0.0001  # Learning rate
		self.optimizer_kwargs = {}

		self.w_decay = 0.0                      # Weight of l2 norm loss

		self.ae_drop_out_rate = 0.4  # Dropout rate of autoencoder
		self.ae_units = [1024, 512, 256]  # Units of Autoencoder: n_features*1024 - relu - 1024*512 - relu - 512*256 - relu - 256*512 - relu - 512*1024 - relu - 1024*n_features - relu
		self.ae_acts = [RELU, RELU, RELU]

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


class AEModel(AIDEModel):
	def __init__(self, mode, config, batch):
		super(AEModel, self).__init__(mode, config, batch)


	def forward(self, batch):
		X = batch
		c = self.config

		if type(X) == tf.SparseTensor:
			X = tf.sparse_tensor_to_dense(X, validate_indices=False)

		if X.get_shape().as_list()[-1] is None:
			X = tf.reshape(X, (-1, c.n_features))

		# encoder
		with tf.variable_scope('AE'):
			self.ae_h = self.encoder(X, c.ae_units, c.ae_acts, c.ae_drop_out_rate)
			units, acts = self.get_decoder_acts_units(c.ae_units, c.ae_acts, c.n_features)
			X_hat = self.decoder(self.ae_h, units, acts)

		if self.mode == PREDICT_MODE:
			return

		# loss
		self.reconstruct_loss = self.mds_loss = self.l2_loss = tf.constant(0., dtype=tf.float32)
		self.reconstruct_loss = self.get_reconstruct_loss(X, X_hat)

		if c.w_decay > 1e-8:   # l2 loss
			self.l2_loss = self.get_l2_loss(c.w_decay)

		self.loss = self.reconstruct_loss + self.l2_loss
		self.all_loss = [self.reconstruct_loss, self.mds_loss, self.l2_loss]

		if self.mode == EVAL_MODE:
			return

		# optimize
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		optimizer = get_optimizer(c.optimizer, c.lr, **c.optimizer_kwargs)
		self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
		self.init_op = tf.global_variables_initializer()


class AE(AIDE):
	def __init__(self, name=None, save_folder=None):
		super(AE, self).__init__(name or 'AE', save_folder)


	def pretrain(self, sess, logger, c):
		pass


	def build(self, config):
		with tf.name_scope(TRAIN_MODE):
			with tf.variable_scope('Model'):
				self.train_data, self.train_data_init_op, self.train_data_ph = self.get_train_data(config)
				self.train_model = AEModel(TRAIN_MODE, config, self.train_data)
		with tf.name_scope(EVAL_MODE):
			with tf.variable_scope('Model', reuse=True):
				self.eval_data, self.eval_data_init_op, self.eval_data_ph = self.get_eval_data(config)
				self.eval_model = AEModel(EVAL_MODE, config, self.eval_data)
		with tf.name_scope(PREDICT_MODE):
			with tf.variable_scope('Model', reuse=True):
				self.pred_data, self.pred_data_init_op, self.pred_data_ph = self.get_predict_data(config)
				self.pred_model = AEModel(PREDICT_MODE, config, self.pred_data)


	def get_embedding(self, sess=None):
		"""
		Args:
			sess (tf.Session)
		Returns:
			np.ndarray: (cell_num, embed_size)
		"""
		sess = sess or self.sess
		sess.run(self.pred_data_init_op, feed_dict=self.get_feed_dict(self.pred_data_ph, self.pred_feed))
		embed_list = []
		try:
			while True:
				embed_list.append(sess.run(self.pred_model.ae_h))
		except tf.errors.OutOfRangeError:
			pass
		return np.vstack(embed_list)