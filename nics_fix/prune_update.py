# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np
import json

__all__ = ["apply_gradients_prune"]

def create_name_no_mask(data):
	basename = data.replace("/","_")
	basename = basename.replace(":","_")
	return basename

def load(mask_file):
	with open(mask_file) as json_config:
		data = json.load(json_config)
		return data

def apply_gradients_prune(origin_func, grads_and_vars, mask_file, **kwargs):
	mask_array = tf.get_default_graph().get_collection("mask")
	mask_data = load(mask_file)
	if not mask_array:
		for (k, v) in mask_data.iteritems():
			mask_name = create_name_no_mask(k)
			mask_value = v[0].strip("[]").split(",")
			mask_np = np.array([1 for x in mask_value])
			for index in range(len(mask_value)):
				mask_np[index] = float(mask_value[index])
			mask_shape = v[1].strip("[]").split(",")
			for index in range(len(mask_shape)):
				mask_shape[index] = int(mask_shape[index])
			mask_shape = tuple(mask_shape)
			mask_new_np = mask_np.reshape(mask_shape)
			mask_var = tf.Variable(mask_new_np, dtype = tf.float32, trainable = False, name = mask_name)
			#tf.variables_initializer([mask_var]).run()
			tf.add_to_collection("mask", mask_var)
		mask_array = tf.get_default_graph().get_collection("mask")
	"""
	else:
		for (k, v) in mask_data.iteritems():
			mask_name = create_name_no_mask(k)
			mask_value = v[0].strip("[]").split(",")
			mask_np = np.array([1 for x in mask_value])
			for index in range(len(mask_value)):
				mask_np[index] = float(mask_value[index])
			mask_shape = v[1].strip("[]").split(",")
			for index in range(len(mask_shape)):
				mask_shape[index] = int(mask_shape[index])
			mask_shape = tuple(mask_shape)
			mask_new_np = mask_np.reshape(mask_shape)
			mask_var_new = tf.Variable(mask_new_np, dtype = tf.float32, trainable = False)
			var = None
			for m in mask_array:
				print(m.name.split(":")[0], "vs", mask_name)
				if m.name.split(":")[0] == mask_name:
					var = m
					break
			assign_op = var.assign(mask_var_new)
			tf.variables_initializer([mask_var_new]).run()
			tf.get_default_session().run(assign_op)
	"""
	print("mask_array: ",mask_array)
	for index in range(len(grads_and_vars)):
		mask_name = create_name_no_mask(grads_and_vars[index][1].name)
		print("loop mask:",mask_name)
		for mask in mask_array:
			if mask.name.split(":")[0] == mask_name:
				print("match:",mask.name)
				new_grad = grads_and_vars[index][0] * mask;
				grads_and_vars[index] = tuple([new_grad, grads_and_vars[index][1]])
	return origin_func(grads_and_vars, **kwargs)