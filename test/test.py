"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import scipy.sparse as sp
import numpy as np
from aide import AIDE, AIDEConfig
from aide.utils_tf import write_csr_to_tfrecord, write_csr_shards_to_tfrecord, write_ary_to_tfrecord, write_ary_shards_to_tfrecord


def get_rand_csr(n_samples=1000, n_features=2000, dtype=np.float32):
	"""
	Args:
		n_samples (int)
		n_features (int)
	Returns:
		sp.csr_matrix: shape=(n_samples, n_features)
	"""
	X = np.random.rand(n_samples, n_features)
	X[X < 0.5] = 0.
	return sp.csr_matrix(X).astype(dtype)


def get_rand_ary(n_samples=1000, n_features=2000, dtype=np.float32):
	"""
	Args:
		n_samples (int)
		n_features (int)
	Returns:
		np.ndarray: shape=(n_samples, n_features)
	"""
	return np.random.rand(n_samples, n_features).astype(dtype)


def get_large_rand_csr_tfrecord(n_samples=100000, n_features=2000, dtype=np.float32):
	"""
	Args:
		n_samples (int)
		n_features (int)
	Returns:
		str: tfrecord data folder for training
		str: tfrecord data path for prediction (finally generating embedding)
	"""
	X = get_rand_csr(n_samples, n_features, dtype)
	train_data_folder = f'train_csr_{n_samples}_{n_features}_{dtype.__name__}_shards'
	pred_data_path = f'pred_csr_{n_samples}_{n_features}_{dtype.__name__}.tfrecord'
	if not (os.path.exists(train_data_folder) and os.path.exists(pred_data_path)):
		write_csr_shards_to_tfrecord(X, tf_folder=train_data_folder, shard_num=10, shuffle=True)
		write_csr_to_tfrecord(X, tf_path=pred_data_path, shuffle=False)
	info_dict = {'n_samples': n_samples, 'n_features': n_features, 'issparse': True}
	return (train_data_folder, pred_data_path), info_dict


def get_large_rand_ary_tfrecord(n_samples=100000, n_features=2000, dtype=np.float32):
	"""
		Args:
			n_samples (int)
			n_features (int)
		Returns:
			str: tfrecord data folder for training
			str: tfrecord data path for prediction (finally generating embedding)
			dict: {}
	"""
	X = get_rand_ary(n_samples, n_features, dtype)
	train_data_folder = f'train_ary_{n_samples}_{n_features}_{dtype.__name__}_shards'
	pred_data_path = f'pred_ary_{n_samples}_{n_features}_{dtype.__name__}.tfrecord'
	if not (os.path.exists(train_data_folder) and os.path.exists(pred_data_path)):
		write_ary_shards_to_tfrecord(X, tf_folder=train_data_folder, shard_num=10, shuffle=True)
		write_ary_to_tfrecord(X, tf_path=pred_data_path, shuffle=False)
	info_dict = {'n_samples': n_samples, 'n_features': n_features, 'issparse': False}
	return (train_data_folder, pred_data_path), info_dict


def test_transform(n_samples, n_features, dtype, issparse, input_tfrecord):
	encoder_name = f'aide_{n_samples}_{n_features}_{dtype.__name__}_{issparse}_{input_tfrecord}'

	if input_tfrecord:
		if issparse:
			X = get_large_rand_csr_tfrecord(n_samples, n_features, dtype)
		else:
			X = get_large_rand_ary_tfrecord(n_samples, n_features, dtype)
	else:
		if issparse:
			X = get_rand_csr(n_samples, n_features, dtype)
		else:
			X = get_rand_ary(n_samples, n_features, dtype)

	config = AIDEConfig(); config.max_step_num = 6000
	encoder = AIDE(name=encoder_name, save_folder=encoder_name)
	embedding = encoder.fit_transform(X, config)  # np.ndarray; (1000, 256)

	assert isinstance(embedding, np.ndarray)
	if input_tfrecord:
		assert embedding.shape == (config.n_samples, config.mds_units[-1])
		assert embedding.dtype.name == config.dtype
	else:
		assert embedding.shape == (X.shape[0], config.mds_units[-1])
		assert embedding.dtype == X.dtype


if __name__ == '__main__':
	test_transform(1000, 2000, dtype=np.float32, issparse=True, input_tfrecord=False)
	test_transform(1000, 2000, dtype=np.float64, issparse=True, input_tfrecord=False)
	test_transform(1000, 2000, dtype=np.float32, issparse=False, input_tfrecord=False)
	test_transform(1000, 2000, dtype=np.float64, issparse=False, input_tfrecord=False)

	# Note: only float32 is supported
	test_transform(100000, 2000, dtype=np.float32, issparse=True, input_tfrecord=True)
	test_transform(100000, 2000, dtype=np.float32, issparse=False, input_tfrecord=True)

