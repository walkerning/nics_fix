# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np
import json, copy

__all__ = ["prune", "auto_prune_all", "count_zero", "prune_all"]

def store(data, file_name='mask.json'):
    with open(file_name, 'w') as json_config:
        json_config.write(json.dumps(data))

def prune(var, ratio, given_dimenson = [1,1,1,1]):
    dimenson = copy.deepcopy(given_dimenson)
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
                prune_shape[index] = var.shape.as_list()[index]
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
    mask_var = tf.Variable(new_mask_array, dtype = tf.float32, trainable = False)
    assign_op = var.assign(tf.multiply(var, mask_var))
    tf.variables_initializer([mask_var]).run()
    tf.get_default_session().run(assign_op)
    return var.name, list(mask_var.eval().flatten())

def auto_prune_all(accuracy_func, thres=0.01, dimenson=[1,1,1,1], mask_file='mask.json', result_pos=0):
    origin_accu = accuracy_func()[result_pos]
    print("origin_accu:",origin_accu)
    var_array = tf.get_default_graph().get_collection("trainable_variables")
    saver = tf.train.Saver(var_array)
    saver.save(tf.get_default_session(), "backup_for_prune")
    ratio_dict = {}
    for var in var_array:
        if len(var.shape.as_list()) not in [2,4]:
            continue
        for ratio in range(9, 0, -1):
            saver.restore(tf.get_default_session(), "backup_for_prune")
            prune(var, ratio/10., dimenson)
            result = accuracy_func()[result_pos]
            print("result: " + str(result) + " for ratio " + str(ratio / 10.) + ", origin: " + str(origin_accu) + " gap: " + str(origin_accu - result) + ",thresh: " + str(thres))
            if origin_accu - result < thres:
                ratio_dict[var] = ratio/10.
                break
    saver.restore(tf.get_default_session(), "backup_for_prune")
    output_dict = {}
    for (k,v) in ratio_dict.iteritems():
        print("Finally prune var " + k.name + " by ratio " + str(v))
        mask_name, mask_value = prune(k, v, dimenson)
        output_dict[mask_name] = [str(mask_value), str(k.shape.as_list())]
    store(output_dict, mask_file)

def prune_all(ratio_dict, dimenson=[1,1,1,1], mask_file='mask.json', factor=1.0):
    var_array = tf.get_default_graph().get_collection("trainable_variables")
    output_dict = {}
    for (k,v) in ratio_dict.iteritems():
        print("Finally prune var " + k + " by ratio " + str(v))
        k_var = None
        for var in var_array:
            if var.name == k:
                k_var = var
                break
        mask_name, mask_value = prune(k_var, v*factor, dimenson)
        output_dict[mask_name] = [str(mask_value), str(k_var.shape.as_list())]
    store(output_dict, mask_file)

def count_zero():
    var_array = tf.get_default_graph().get_collection("trainable_variables")
    all_count = 0
    all_total = 0
    for var in var_array:
        count = 0
        index = 0
        value_array = var.eval().flatten()
        for value in value_array:
            if abs(value) < 1e-6:
                count += 1
                all_count += 1
            index += 1
        print("var: ", var.name, " zero: ", str(count), " total: ", str(reduce(lambda x,y:x*y, var.shape.as_list())), " ratio: ", str(float(count)/reduce(lambda x,y:x*y, var.shape.as_list())))
        all_total += reduce(lambda x,y:x*y, var.shape.as_list())
    print("[All] zero: ", str(all_count), " total: ", str(all_total), " ratio: ", str(float(all_count)/all_total))