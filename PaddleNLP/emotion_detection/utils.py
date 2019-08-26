"""
EmoTect utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import random

import paddle
import paddle.fluid as fluid
import numpy as np


def init_inference_model(exe, inference_model_path):
    assert isinstance(inference_model_path, str)

    if not os.path.exists(inference_model_path):
        raise Warning("The inference model path do not exist.")
        return False

    [inference_program, feed_names, fetch_targets] = fluid.io.load_inference_model(
            dirname=inference_model_path,
            executor=exe,
            model_filename="model.pdmodel",
            params_filename="params.pdparams")
    return inference_program


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """
    Init CheckPoint
    """
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        """
        If existed presitabels
        """
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))


def data_reader(file_path, word_dict, num_examples, phrase, epoch=1):
    """
    Convert word sequence into slot
    """
    unk_id = len(word_dict)
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            if line.startswith("label"):
                continue
            if phrase == "infer":
                cols = line.strip().split("\t")
                if len(cols) != 1:
                    query = cols[-1]
                wids = [word_dict[x] if x in word_dict else unk_id
                        for x in query.strip().split(" ")]
                all_data.append((wids,))
            else:
                cols = line.strip().split("\t")
                if len(cols) != 2:
                    sys.stderr.write("[NOTICE] Error Format Line!")
                    continue
                label = int(cols[0])
                wids = [word_dict[x] if x in word_dict else unk_id
                        for x in cols[1].split(" ")]
                all_data.append((wids, label))
    num_examples[phrase] = len(all_data)

    if phrase == "infer":
        def reader():
            """
            Infer reader function
            """
            for wids in all_data:
                yield wids
        return reader

    def reader():
        """
        Reader function
        """
        for idx in range(epoch):
            if phrase == "train":
                random.shuffle(all_data)
            for wids, label in all_data:
                yield wids, label
    return reader


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as fin:
        wid = 0
        for line in fin:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab

def to_lodtensor(data, place):
    """
    convert ot LODtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def data2tensor(data, place):
    input_seq = to_lodtensor(list(map(lambda x: x[0], data)), place)
    return input_seq

def prepare_data(data_path, vocab_path, batch_size):
    sample_num = {'infer':-1}
    assert os.path.exists(
        data_path), "The given data does not exist."
    assert os.path.exists(
        vocab_path), "The given word dictionary dose not exist."
    vocab = load_vocab(vocab_path)
    reader = paddle.batch(data_reader(data_path, vocab, sample_num, 'infer'),
                          batch_size)
    return reader
