# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np

__all__ = ["prune"]

def prune(ratio, name, dimenson = [1,1,1,1]):
	tensor_array = tf.get_default_graph().get_collection("trainable_variables")
	for tensor in tensor_array:
		if tensor.name != name:
			continue
		if tensor.shape.as_list()[0] > 0:
			prune_shape = None
			new_shape = None
			if len(tensor.shape.as_list()) == 2:
				prune_shape = [1,1,1,1]
				new_shape = [1,1] + tensor.shape.as_list()
			else:
				if len(dimenson) != 4:
					print("error: incorrect prune dimenson")
					exit(0)
				prune_shape = dimenson
				for index in range(len(prune_shape)):
					if prune_shape[index] == 0:
						prune_shape[index] = tensor.shape[index]
				new_shape = tensor.shape.as_list()
			print("prune layer: " + name + " ratio: " + str(ratio) + " dimen: " + str(prune_shape) + " shape: " + str(new_shape))
			data_array = tensor.eval().flatten()
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
			new_mask_array = mask_array.reshape(tensor.eval().shape)
			mask_var = tf.Variable(new_mask_array, dtype = tf.float32)
			tensor = tensor * mask_var
			tf.variables_initializer([mask_var]).run()
			tf.get_default_session().run(tensor)