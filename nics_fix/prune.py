# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np

__all__ = ["prune", "auto_prune_all"]

def prune(var, ratio, dimenson = [1,1,1,1]):
	prune_shape = None
	new_shape = None
	if len(var.shape.as_list()) == 2:
		prune_shape = [1,1,1,1]
		new_shape = [1,1] + var.shape.as_list()
	else:
		if len(dimenson) != 4:
			print("error: incorrect prune dimenson")
			exit(0)
		prune_shape = dimenson
		for index in range(len(prune_shape)):
			if prune_shape[index] == 0:
				prune_shape[index] = var.shape[index]
		new_shape = var.shape.as_list()
	print("prune layer: " + var.name + " ratio: " + str(ratio) + " dimen: " + str(prune_shape) + " shape: " + str(new_shape))
	data_array = var.eval().flatten()
	mult_func = lambda x,y:x*y
	for i in range(0, new_shape[0], prune_shape[0]):
		for j in range(0, new_shape[1], prune_shape[1]):
			for k in range(0, new_shape[2], prune_shape[2]):
				for l in range(0, new_shape[3], prune_shape[3]):
					new_value = 0
					for i_p in range(i, i + prune_shape[0]):
						for j_p in range(j, j + prune_shape[1]):
							for k_p in range(k, k + prune_shape[2]):
								for l_p in range(l, l + prune_shape[3]):
									new_value += data_array[i_p * reduce(mult_func, new_shape[1:]) \
											+ j_p * reduce(mult_func, new_shape[2:]) + k_p * new_shape[3] + l_p] ** 2
					for i_p in range(i, i + prune_shape[0]):
						for j_p in range(j, j + prune_shape[1]):
							for k_p in range(k, k + prune_shape[2]):
								for l_p in range(l, l + prune_shape[3]):
									data_array[i_p * reduce(mult_func, new_shape[1:]) \
											+ j_p * reduce(mult_func, new_shape[2:]) + k_p * new_shape[3] + l_p] = new_value
	mask_array = np.array([1 for x in data_array])
	data_index = range(len(data_array))
	data_dict = dict(zip(data_index, data_array))
	new_dict = sorted(data_dict.iteritems(), key = lambda x:x[1])
	part_size = reduce(mult_func, prune_shape)
	prune_num = np.floor(ratio * (len(data_array) / part_size)) * part_size
	iter_index = 0
	for (k,v) in new_dict:
		mask_array[k] = 0
		iter_index += 1
		if iter_index >= prune_num:
			break
	new_mask_array = mask_array.reshape(var.eval().shape)
	mask_var = tf.Variable(new_mask_array, dtype = tf.float32)
	assign_op = var.assign(tf.multiply(var, mask_var))
	tf.variables_initializer([mask_var]).run()
	tf.get_default_session().run(assign_op)

def auto_prune_all(op, op_dict, thres=0.01, dimenson=[1,1,1,1]):
	origin_accu = tf.get_default_session().run(op, feed_dict=op_dict)
	var_array = tf.get_default_graph().get_collection("trainable_variables")
	saver = tf.train.Saver(var_array)
	saver.save(tf.get_default_session(), "backup_for_prune")
	ratio_dict = {}
	for var in var_array:
		if len(var.shape.as_list()) not in [2,4]:
			continue
		for ratio in range(9, 0, -1):
			prune(var, ratio/10., dimenson)
			result = tf.get_default_session().run(op, feed_dict=op_dict)
			print("result: " + str(result) + " for ratio " + str(ratio / 10.))
			if origin_accu - result < thres:
				ratio_dict[var] = ratio/10.
				break
	saver.restore(tf.get_default_session(), "backup_for_prune")
	for (k,v) in ratio_dict.iteritems():
		print("Finally prune var " + k.name + " by ratio " + str(v))
		prune(k, v, dimenson)