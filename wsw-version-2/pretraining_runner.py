# coding=utf-8
"""Run (SUI + ESR + RUND + SUNP) TF2.0 examples for Who-Says-What (WSW)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from time import time

import tensorflow as tf

import modeling_speaker as modeling
import optimization

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string("task_name", 'WSW Pre-training',
                    "The name of the task to train.")

flags.DEFINE_string("input_file", './data/ijcai2019/pretraining_data.tfrecord',
                    "The input data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("output_dir", './uncased_L-12_H-768_A-12_pretrained',
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("bert_config_file", './plm/uncased_L-12_H-768_A-12/bert_config.json',
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", './plm/uncased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("init_checkpoint", './plm/uncased_L-12_H-768_A-12/bert_model.ckpt',
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_train", True,
                  "Whether to run training.")

flags.DEFINE_integer("mid_save_step", 20000,
                     "Epoch is so long, mid_save_step 15000 is roughly 3 hours")

flags.DEFINE_integer("max_seq_length", 230,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_utr_length", 30,
                     "Maximum single utterance length.")

flags.DEFINE_integer("max_utr_num", 7,
                     "Maximum utterance number.")

flags.DEFINE_integer("max_predictions_per_seq", 25,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq_sui", 4,
                     "Maximum number of Speaker Utterance Identification (SUI) predictions per sequence.")

flags.DEFINE_integer("max_predictions_per_seq_esr", 2,
                     "Maximum number of Exact Speaker Recognition (ESR) predictions per sequence.")

flags.DEFINE_integer("train_batch_size", 4,
                     "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("warmup_proportion", 0.1,
                   "Number of warmup steps.")

flags.DEFINE_integer("num_train_epochs", 10,
                     "num_train_epochs.")


def print_configuration_op(FLAGS):
    print('My Configurations:')
    for name, value in FLAGS.__flags.items():
        value = value.value
        if type(value) == float:
            print(' %s:\t %f' % (name, value))
        elif type(value) == int:
            print(' %s:\t %d' % (name, value))
        elif type(value) == str:
            print(' %s:\t %s' % (name, value))
        elif type(value) == bool:
            print(' %s:\t %s' % (name, value))
        else:
            print('%s:\t %s' % (name, value))
    print('End of configuration')


def count_data_size(file_name):
    sample_nums = 0
    for _ in tf.compat.v1.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return sample_nums


def parse_exmp(serial_exmp):
    input_data = tf.compat.v1.parse_single_example(serial_exmp,
                                                   features={
                                                       "input_ids_mlm_nsp":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "input_mask_mlm_nsp":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "segment_ids_mlm_nsp":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "speaker_ids_mlm_nsp":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "next_sentence_labels":
                                                           tf.compat.v1.FixedLenFeature([1], tf.int64),
                                                       "masked_lm_positions":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_predictions_per_seq],
                                                                                        tf.int64),
                                                       "masked_lm_ids":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_predictions_per_seq],
                                                                                        tf.int64),
                                                       "masked_lm_weights":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_predictions_per_seq],
                                                                                        tf.float32),

                                                       "cls_positions":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_utr_num], tf.int64),
                                                       "input_ids_sui_esr":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "input_mask_sui_esr":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "segment_ids_sui_esr":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "speaker_ids_sui_esr":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),

                                                       "sui_recog_positions":
                                                           tf.compat.v1.FixedLenFeature(
                                                               [FLAGS.max_predictions_per_seq_ar],
                                                               tf.int64),
                                                       "sui_recog_labels":
                                                           tf.compat.v1.FixedLenFeature(
                                                               [FLAGS.max_predictions_per_seq_ar],
                                                               tf.int64),
                                                       "sui_recog_weights":
                                                           tf.compat.v1.FixedLenFeature(
                                                               [FLAGS.max_predictions_per_seq_ar],
                                                               tf.float32),

                                                       "exact_sr_positions":
                                                           tf.compat.v1.FixedLenFeature(
                                                               [FLAGS.max_predictions_per_seq_sr],
                                                               tf.int64),
                                                       "exact_sr_labels":
                                                           tf.compat.v1.FixedLenFeature(
                                                               [FLAGS.max_predictions_per_seq_sr],
                                                               tf.int64),
                                                       "exact_sr_weights":
                                                           tf.compat.v1.FixedLenFeature(
                                                               [FLAGS.max_predictions_per_seq_sr],
                                                               tf.float32),
                                                       "input_ids_sunp":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "input_mask_sunp":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "segment_ids_sunp":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "speaker_ids_sunp":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_seq_length],
                                                                                        tf.int64),
                                                       "root_und_positions":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_utr_length],
                                                                                        tf.int64),
                                                       "root_und_ids":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_utr_length],
                                                                                        tf.int64),
                                                       "root_und_weights":
                                                           tf.compat.v1.FixedLenFeature([FLAGS.max_utr_length],
                                                                                        tf.float32),
                                                       "next_thread_labels":
                                                           tf.compat.v1.FixedLenFeature([1], tf.int64),
                                                   }
                                                   )
    # So cast all int64 to int32.
    for name in list(input_data.keys()):
        t = input_data[name]
        if t.dtype == tf.int64:
            t = tf.compat.v1.to_int32(t)
        input_data[name] = t

    input_ids_mlm_nsp = input_data["input_ids_mlm_nsp"]
    input_mask_mlm_nsp = input_data["input_mask_mlm_nsp"]
    segment_ids_mlm_nsp = input_data["segment_ids_mlm_nsp"]
    speaker_ids_mlm_nsp = input_data["speaker_ids_mlm_nsp"]
    next_sentence_labels = input_data["next_sentence_labels"]
    masked_lm_positions = input_data["masked_lm_positions"]
    masked_lm_ids = input_data["masked_lm_ids"]
    masked_lm_weights = input_data["masked_lm_weights"]

    cls_positions = input_data["cls_positions"]
    input_ids_sui_esr = input_data["input_ids_sui_esr"]
    input_mask_sui_esr = input_data["input_mask_sui_esr"]
    segment_ids_sui_esr = input_data["segment_ids_sui_esr"]
    speaker_ids_sui_esr = input_data["speaker_ids_sui_esr"]

    sui_recog_positions = input_data["sui_recog_positions"]
    sui_recog_labels = input_data["sui_recog_labels"]
    sui_recog_weights = input_data["sui_recog_weights"]

    exact_sr_positions = input_data["exact_sr_positions"]
    exact_sr_labels = input_data["exact_sr_labels"]
    exact_sr_weights = input_data["exact_sr_weights"]

    input_ids_sunp = input_data["input_ids_sunp"]
    input_mask_sunp = input_data["input_mask_sunp"]
    segment_ids_sunp = input_data["segment_ids_sunp"]
    speaker_ids_sunp = input_data["speaker_ids_sunp"]

    root_und_positions = input_data["root_und_positions"]
    root_und_ids = input_data["root_und_ids"]
    root_und_weights = input_data["root_und_weights"]
    next_thread_labels = input_data["next_thread_labels"]

    return input_ids_mlm_nsp, input_mask_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp, \
        next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
        cls_positions, input_ids_sui_esr, input_mask_sui_esr, segment_ids_sui_esr, speaker_ids_sui_esr, \
        sui_recog_positions, sui_recog_labels, sui_recog_weights, exact_sr_positions, exact_sr_labels, exact_sr_weights, \
        input_ids_sunp, input_mask_sunp, segment_ids_sunp, speaker_ids_sunp, root_und_positions, root_und_ids, \
        root_und_weights, next_thread_labels


def model_fn_builder(features, is_training, bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    (input_ids_mlm_nsp, input_mask_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp,
     next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights,
     cls_positions, input_ids_sui_esr, input_mask_sui_esr, segment_ids_sui_esr, speaker_ids_sui_esr,
     sui_recog_positions, sui_recog_labels, sui_recog_weights, exact_sr_positions, exact_sr_labels, exact_sr_weights,
     input_ids_sunp, input_mask_sunp, segment_ids_sunp, speaker_ids_sunp,
     shared_unp_positions, shared_unp_ids, shared_unp_weights, input_ids_rund, input_mask_rund, segment_ids_rund,
     speaker_ids_rund, next_thread_labels) = features

    model_mlm_nsp = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_mlm_nsp,
        input_mask=input_mask_mlm_nsp,
        token_type_ids=segment_ids_mlm_nsp,
        speaker_ids=speaker_ids_mlm_nsp,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        scope_reuse=False)

    (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
        bert_config, model_mlm_nsp.get_sequence_output(), model_mlm_nsp.get_embedding_table(), masked_lm_positions,
        masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output(
        bert_config, model_mlm_nsp.get_pooled_output(), next_sentence_labels)

    model_sui_esr = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_sui_esr,
        input_mask=input_mask_sui_esr,
        token_type_ids=segment_ids_sui_esr,
        speaker_ids=speaker_ids_sui_esr,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        scope_reuse=True)

    (sui_recog_loss, sui_recog_example_loss, sui_recog_log_probs) = get_speaker_utterance_identification_output(
        bert_config, model_sui_esr.get_sequence_output(), cls_positions, sui_recog_positions, sui_recog_labels,
        sui_recog_weights)

    (masked_esr_loss, masked_esr_example_loss, masked_esr_log_probs) = get_exact_speaker_recognition_output(
        bert_config, model_sui_esr.get_sequence_output(), cls_positions, exact_sr_positions, exact_sr_labels,
        exact_sr_weights)

    model_sunp = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_sunp,
        input_mask=input_mask_sunp,
        token_type_ids=segment_ids_sunp,
        speaker_ids=speaker_ids_sunp,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        scope_reuse=True)

    (shared_unp_loss, shared_unp_example_loss, shared_unp_log_probs) = get_shared_utterance_node_predictions_output(
        bert_config, model_sunp.get_sequence_output(), model_sunp.get_embedding_table(), shared_unp_positions,
        shared_unp_ids, shared_unp_weights)

    model_rund = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids_rund,
        input_mask=input_mask_rund,
        token_type_ids=segment_ids_rund,
        speaker_ids=speaker_ids_rund,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        scope_reuse=True)

    (root_und_loss, root_und_example_loss, root_und_log_probs) = get_root_utterance_node_detection_output(
        bert_config, model_rund.get_pooled_output(), next_thread_labels)

    total_loss = masked_lm_loss + next_sentence_loss + \
                 sui_recog_loss + masked_esr_loss + \
                 shared_unp_loss + root_und_loss

    tvars = tf.compat.v1.trainable_variables()
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

    metrics = metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights,
                        next_sentence_example_loss, next_sentence_log_probs, next_sentence_labels,
                        sui_recog_example_loss, sui_recog_log_probs, sui_recog_labels, sui_recog_weights,
                        masked_esr_example_loss, masked_esr_log_probs, shared_unp_example_loss, shared_unp_log_probs,
                        root_und_example_loss, root_und_log_probs, exact_sr_labels, exact_sr_weights,
                        shared_unp_ids, shared_unp_weights)

    return train_op, total_loss, metrics, input_ids_mlm_nsp


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)  # [batch_size*max_predictions_per_seq, dim]

    with tf.compat.v1.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        with tf.compat.v1.variable_scope("transform"):
            input_tensor = tf.compat.v1.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.compat.v1.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.compat.v1.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)  # [batch_size*max_predictions_per_seq, vocab_size]

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])
        one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels,
                                          axis=[-1])  # [batch_size*max_predictions_per_seq, ]
        numerator = tf.reduce_sum(label_weights * per_example_loss)  # [1, ]
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return loss, per_example_loss, log_probs


def get_next_sentence_output(bert_config, input_tensor, labels):
    with tf.compat.v1.variable_scope("cls/seq_relationship"):
        output_weights = tf.compat.v1.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.compat.v1.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.compat.v1.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)  # [batch_size, 2]
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, log_probs


def get_speaker_utterance_identification_output(bert_config, input_tensor, cls_positions, sui_recog_positions,
                                                sui_recog_labels, sui_recog_weights):
    cls_shape = modeling.get_shape_list(cls_positions, expected_rank=2)
    max_utr_num = cls_shape[1]

    input_tensor = gather_indexes(input_tensor, cls_positions)  # [batch_size*max_utr_num, dim]
    input_shape = modeling.get_shape_list(input_tensor, expected_rank=2)
    width = input_shape[1]

    with tf.compat.v1.variable_scope("cls/addressee_recognize"):
        # We apply one more non-linear transformation before the output layer.
        with tf.compat.v1.variable_scope("transform"):
            input_tensor = tf.compat.v1.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)  # [batch_size*max_utr_num, dim]

        input_tensor = tf.compat.v1.reshape(input_tensor, [-1, max_utr_num, width])  # [batch_size, max_utr_num, dim]
        output_weights = tf.compat.v1.get_variable(
            "output_weights",
            shape=[width, width],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        logits = tf.compat.v1.matmul(tf.einsum('aij,jk->aik', input_tensor, output_weights),
                                     input_tensor, transpose_b=True)  # [batch_size, max_utr_num, max_utr_num]

        # mask = [[0. 0. 0. 0. 0.]
        #         [1. 0. 0. 0. 0.]
        #         [1. 1. 0. 0. 0.]
        #         [1. 1. 1. 0. 0.]
        #         [1. 1. 1. 1. 0.]]
        mask = tf.compat.v1.matrix_band_part(tf.ones((max_utr_num, max_utr_num)), -1,
                                             0) - tf.compat.v1.matrix_band_part(
            tf.ones((max_utr_num, max_utr_num)), 0, 0)
        logits = logits * mask + -1e9 * (1 - mask)  # [batch_size, max_utr_num, max_utr_num]

        logits = gather_indexes(logits, sui_recog_positions)  # [batch_size*max_predictions_per_seq_ar, max_utr_num]
        log_probs = tf.nn.log_softmax(logits, axis=-1)  # [batch_size*max_predictions_per_seq_ar, max_utr_num]

        label_ids = tf.reshape(sui_recog_labels, [-1])  # [batch_size*max_predictions_per_seq_ar, ]
        label_weights = tf.reshape(sui_recog_weights, [-1])  # [batch_size*max_predictions_per_seq_ar, ]
        one_hot_labels = tf.one_hot(label_ids, depth=max_utr_num,
                                    dtype=tf.float32)  # [batch_size*max_predictions_per_seq_ar, max_utr_num]

        # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
        per_example_loss = - tf.reduce_sum(log_probs * one_hot_labels,
                                           axis=[-1])  # [batch_size*max_predictions_per_seq_ar, ]
        numerator = tf.reduce_sum(label_weights * per_example_loss)  # [1, ]
        denominator = tf.reduce_sum(label_weights) + 1e-5  # [1, ]
        loss = numerator / denominator

        return loss, per_example_loss, log_probs


def get_exact_speaker_recognition_output(bert_config, input_tensor, cls_positions, masked_esr_positions,
                                         masked_esr_labels, masked_esr_weights):
    cls_shape = modeling.get_shape_list(cls_positions, expected_rank=2)
    max_utr_num = cls_shape[1]

    input_tensor = gather_indexes(input_tensor, cls_positions)  # [batch_size*max_utr_num, dim]
    input_shape = modeling.get_shape_list(input_tensor, expected_rank=2)
    width = input_shape[1]

    with tf.compat.v1.variable_scope("cls/speaker_restore"):
        # We apply one more non-linear transformation before the output layer.
        with tf.compat.v1.variable_scope("transform"):
            input_tensor = tf.compat.v1.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)  # [batch_size*max_utr_num, dim]

        input_tensor = tf.reshape(input_tensor, [-1, max_utr_num, width])  # [batch_size, max_utr_num, dim]
        output_weights = tf.compat.v1.get_variable(
            "output_weights",
            shape=[width, width],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        logits = tf.compat.v1.matmul(tf.einsum('aij,jk->aik', input_tensor, output_weights),
                                     input_tensor, transpose_b=True)  # [batch_size, max_utr_num, max_utr_num]

        # mask = [[0. 0. 0. 0. 0.]
        #         [1. 0. 0. 0. 0.]
        #         [1. 1. 0. 0. 0.]
        #         [1. 1. 1. 0. 0.]
        #         [1. 1. 1. 1. 0.]]
        mask = tf.compat.v1.matrix_band_part(tf.ones((max_utr_num, max_utr_num)), -1,
                                             0) - tf.compat.v1.matrix_band_part(
            tf.ones((max_utr_num, max_utr_num)), 0, 0)
        logits = logits * mask + -1e9 * (1 - mask)  # [batch_size, max_utr_num, max_utr_num]

        logits = gather_indexes(logits, masked_esr_positions)  # [batch_size*max_predictions_per_seq_sr, max_utr_num]
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(masked_esr_labels, [-1])  # [batch_size*max_predictions_per_seq_sr, ]
        label_weights = tf.reshape(masked_esr_weights, [-1])  # [batch_size*max_predictions_per_seq_sr, ]
        one_hot_labels = tf.one_hot(label_ids, depth=max_utr_num,
                                    dtype=tf.float32)  # [batch_size*max_predictions_per_seq_sr, max_utr_num]

        # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
        per_example_loss = - tf.reduce_sum(log_probs * one_hot_labels,
                                           axis=[-1])  # [batch_size*max_predictions_per_seq_sr, ]
        numerator = tf.reduce_sum(label_weights * per_example_loss)  # [1, ]
        denominator = tf.reduce_sum(label_weights) + 1e-5  # [1, ]
        loss = numerator / denominator

        return loss, per_example_loss, log_probs


def get_shared_utterance_node_predictions_output(bert_config, input_tensor, output_weights, positions, label_ids,
                                                 label_weights):
    """Get loss and log probs for the Shared Utterance Node Predictions (SUNP)."""

    input_tensor = gather_indexes(input_tensor, positions)  # [batch_size*max_utr_length, dim]

    with tf.compat.v1.variable_scope("cls/shared_utterance_restore"):
        # We apply one more non-linear transformation before the output layer.
        with tf.compat.v1.variable_scope("transform"):
            input_tensor = tf.compat.v1.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.compat.v1.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)  # [batch_size*max_utr_length, vocab_size]

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])
        one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `label_weights` tensor has a value of 1.0 for every real prediction and 0.0 for the padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels,
                                          axis=[-1])  # [batch_size*max_predictions_per_seq, ]
        numerator = tf.reduce_sum(label_weights * per_example_loss)  # [1, ]
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return loss, per_example_loss, log_probs


def get_root_utterance_node_detection_output(bert_config, input_tensor, labels):
    with tf.compat.v1.compat.v1.variable_scope("cls/root_utterance_node_detect"):
        output_weights = tf.compat.v1.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.compat.v1.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)  # [batch_size, 2]
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # [batch_size, ]
        loss = tf.reduce_mean(per_example_loss)  # [1, ]

        return loss, per_example_loss, log_probs


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])  # [batch_size, 1]
    flat_positions = tf.reshape(positions + flat_offsets, [-1])  # [batch_size*max_predictions_per_seq, ]
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)  # [batch_size*max_predictions_per_seq, width]
    return output_tensor


def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights,
              next_sentence_example_loss, next_sentence_log_probs, next_sentence_labels,
              adr_recog_example_loss, adr_recog_log_probs, adr_recog_labels, adr_recog_weights,
              masked_sr_example_loss, masked_sr_log_probs, masked_sr_labels, masked_sr_weights,
              shared_nd_example_loss, shared_nd_log_probs, next_thread_labels):
    masked_lm_predictions = tf.compat.v1.argmax(masked_lm_log_probs,
                                                axis=-1, output_type=tf.int32)  # [batch_size*max_predictions_per_seq, ]
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.compat.v1.metrics.mean(
        values=masked_lm_example_loss,
        weights=masked_lm_weights)

    next_sentence_predictions = tf.argmax(next_sentence_log_probs, axis=-1, output_type=tf.int32)  # [batch_size, ]
    next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
        labels=next_sentence_labels,
        predictions=next_sentence_predictions)
    next_sentence_mean_loss = tf.compat.v1.metrics.mean(
        values=next_sentence_example_loss)

    adr_recog_predictions = tf.compat.v1.argmax(adr_recog_log_probs, axis=-1,
                                                output_type=tf.int32)  # [batch_size*max_predictions_per_seq_ar, ]
    adr_recog_labels = tf.reshape(adr_recog_labels, [-1])  # [batch_size*max_predictions_per_seq_ar, ]
    adr_recog_weights = tf.reshape(adr_recog_weights, [-1])  # [batch_size*max_predictions_per_seq_ar, ]
    adr_recog_accuracy = tf.compat.v1.metrics.accuracy(
        labels=adr_recog_labels,
        predictions=adr_recog_predictions,
        weights=adr_recog_weights)
    adr_recog_mean_loss = tf.compat.v1.metrics.mean(
        values=adr_recog_example_loss,
        weights=adr_recog_weights)

    masked_sr_predictions = tf.argmax(masked_sr_log_probs, axis=-1,
                                      output_type=tf.int32)  # [batch_size*max_predictions_per_seq_sr, ]
    masked_sr_labels = tf.reshape(masked_sr_labels, [-1])  # [batch_size*max_predictions_per_seq_sr, ]
    masked_sr_weights = tf.reshape(masked_sr_weights, [-1])  # [batch_size*max_predictions_per_seq_sr, ]
    masked_sr_accuracy = tf.compat.v1.metrics.accuracy(
        labels=masked_sr_labels,
        predictions=masked_sr_predictions,
        weights=masked_sr_weights)
    masked_sr_mean_loss = tf.compat.v1.metrics.mean(
        values=masked_sr_example_loss,
        weights=masked_sr_weights)

    shared_nd_predictions = tf.argmax(shared_nd_log_probs, axis=-1, output_type=tf.int32)  # [batch_size, ]
    shared_nd_accuracy = tf.compat.v1.metrics.accuracy(
        labels=next_thread_labels,
        predictions=shared_nd_predictions)
    shared_nd_mean_loss = tf.compat.v1.metrics.mean(
        values=shared_nd_example_loss)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
        "adr_recog_accuracy": adr_recog_accuracy,
        "adr_recog_loss": adr_recog_mean_loss,
        "masked_sr_accuracy": masked_sr_accuracy,
        "masked_sr_loss": masked_sr_mean_loss,
        "shared_nd_accuracy": shared_nd_accuracy,
        "shared_nd_loss": shared_nd_mean_loss
    }


def run_epoch(epoch, sess, saver, output_dir, epoch_save_step, mid_save_step,
              input_ids, eval_metrics, total_loss, train_op, eval_op):
    total_sample = 0
    step = 0
    t0 = time()

    tf.compat.v1.logging.info("*** Start epoch {} training ***".format(epoch))
    try:
        while True:
            step += 1
            _input_ids, batch_metrics, batch_loss, _, _ = sess.run(
                [input_ids, eval_metrics, total_loss, train_op, eval_op])
            masked_lm_accuracy, masked_lm_loss, next_sentence_accuracy, next_sentence_loss, \
                adr_recog_accuracy, adr_recog_loss, masked_sr_accuracy, masked_sr_loss, \
                pointer_cd_simi, pointer_cd_loss, masked_sur_accuracy, masked_sur_loss, \
                shared_nd_accuracy, shared_nd_loss = batch_metrics

            batch_sample = len(_input_ids)
            total_sample += batch_sample
            # accumulate_loss += batch_loss * batch_sample

            # print
            print_every_step = 200
            if step % print_every_step == 0:
                tf.compat.v1.logging.info("Step: {}, Loss: {:.4f}, Sample: {}, Time (min): {:.2f}".format(
                    step, batch_loss, total_sample, (time() - t0) / 60))
                tf.compat.v1.logging.info(
                    'MLM_accuracy: {:.6f}, MLM_loss: {:.6f}, NSP_accuracy: {:.6f}, NSP_loss: {:.6f}, '
                    'RUR_accuracy: {:.6f}, RUR_loss: {:.6f}, ISS_accuracy: {:.6f}, ISS_loss: {:.6f}, '
                    'PCD_similarity: {:.6f}, PCD_loss: {:.6f}, SUNP_accuracy: {:.6f}, SUNP_loss: {:.6f}, '
                    'SND_accuracy: {:.6f}, SND_loss: {:.6f}'.format(
                        masked_lm_accuracy, masked_lm_loss, next_sentence_accuracy, next_sentence_loss,
                        adr_recog_accuracy, adr_recog_loss, masked_sr_accuracy, masked_sr_loss,
                        pointer_cd_simi, pointer_cd_loss, masked_sur_accuracy, masked_sur_loss,
                        shared_nd_accuracy, shared_nd_loss))

            if (step % mid_save_step == 0) or (step % epoch_save_step == 0):
                # c_time = str(int(time()))
                save_path = os.path.join(output_dir, 'pretrained_bert_model_epoch_{}_step_{}'.format(epoch, step))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                saver.save(sess, os.path.join(save_path, 'bert_model_epoch_{}_step_{}.ckpt'.format(epoch, step)),
                           global_step=step)
                # saver.save(sess, os.path.join(save_path, 'bert_model_epoch_{}_step_{}.ckpt'.format(epoch, step)))
                tf.compat.v1.logging.info('========== Save model at epoch: {}, step: {} =========='.format(epoch, step))
                tf.compat.v1.logging.info("Step: {}, Loss: {:.4f}, Sample: {}, Time (min): {:.2f}".format(
                    step, batch_loss, total_sample, (time() - t0) / 60))
                tf.compat.v1.logging.info(
                    'MLM_accuracy: {:.6f}, MLM_loss: {:.6f}, NSP_accuracy: {:.6f}, NSP_loss: {:.6f}, '
                    'RUR_accuracy: {:.6f}, RUR_loss: {:.6f}, ISS_accuracy: {:.6f}, ISS_loss: {:.6f}, '
                    'PCD_similarity: {:.6f}, PCD_loss: {:.6f}, SUNP_accuracy: {:.6f}, SUNP_loss: {:.6f}, '
                    'SND_accuracy: {:.6f}, SND_loss: {:.6f}'.format(
                        masked_lm_accuracy, masked_lm_loss, next_sentence_accuracy, next_sentence_loss,
                        adr_recog_accuracy, adr_recog_loss, masked_sr_accuracy, masked_sr_loss,
                        pointer_cd_simi, pointer_cd_loss, masked_sur_accuracy, masked_sur_loss,
                        shared_nd_accuracy, shared_nd_loss))

    except tf.errors.OutOfRangeError:
        tf.compat.v1.logging.info('*** Epoch {} is finished ***'.format(epoch))
        pass


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    print_configuration_op(FLAGS)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    train_data_size = count_data_size(FLAGS.input_file)
    tf.compat.v1.logging.info('*** train data size: {} ***'.format(train_data_size))

    num_train_steps = train_data_size // FLAGS.train_batch_size * FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    epoch_save_step = train_data_size // FLAGS.train_batch_size

    buffer_size = 1000
    tf.compat.v1.disable_eager_execution()
    filenames = tf.compat.v1.placeholder(tf.string, shape=[None])
    dataset = tf.compat.v1.data.TFRecordDataset(filenames=[filenames])
    dataset = dataset.map(parse_exmp)
    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(FLAGS.train_batch_size)
    iterator = dataset.make_initializable_iterator()

    (input_ids_mlm_nsp, input_mask_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp, \
        next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
        cls_positions, input_ids_sui_esr, input_mask_sui_esr, segment_ids_sui_esr, speaker_ids_sui_esr, \
        spk_ui_positions, spk_ui_labels, spk_ui_weights, exact_sr_positions, exact_sr_labels, \
        exact_sr_weights, input_ids_sunp, input_mask_sunp, segment_ids_sunp, speaker_ids_sunp,
        root_und_positions, root_und_ids, root_und_weights, next_thread_labels) = iterator.get_next()

    features = [input_ids_mlm_nsp, input_mask_mlm_nsp, segment_ids_mlm_nsp, speaker_ids_mlm_nsp,
                next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights,
                cls_positions, input_ids_sui_esr, input_mask_sui_esr, segment_ids_sui_esr,
                speaker_ids_sui_esr, spk_ui_positions, spk_ui_labels, spk_ui_weights,
                exact_sr_positions, exact_sr_labels, exact_sr_weights, input_ids_sunp, input_mask_sunp,
                segment_ids_sunp, speaker_ids_sunp, root_und_positions, root_und_ids, root_und_weights,
                next_thread_labels]

    train_op, total_loss, metrics, input_ids = model_fn_builder(
        features=features,
        is_training=True,
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    masked_lm_accuracy, masked_lm_accuracy_op = metrics["masked_lm_accuracy"]
    masked_lm_loss, masked_lm_loss_op = metrics["masked_lm_loss"]
    next_sentence_accuracy, next_sentence_op = metrics["next_sentence_accuracy"]
    next_sentence_loss, next_sentence_loss_op = metrics["next_sentence_loss"]
    spk_ui_accuracy, spk_ui_accuracy_op = metrics["spk_ui_accuracy"]
    spk_ui_loss, spk_ui_loss_op = metrics["spk_ui_loss"]
    exact_sr_accuracy, exact_sr_accuracy_op = metrics["exact_sr_accuracy"]
    exact_sr_loss, exact_sr_loss_op = metrics["exact_sr_loss"]
    shared_unp_accuracy, shared_unp_accuracy_op = metrics["shared_unp_accuracy"]
    shared_unp_loss, shared_unp_loss_op = metrics["shared_unp_loss"]
    root_und_accuracy, root_und_accuracy_op = metrics["root_und_accuracy"]
    root_und_loss, root_und_loss_op = metrics["root_und_loss"]

    eval_metrics = [masked_lm_accuracy, masked_lm_loss, next_sentence_accuracy, next_sentence_loss,
                    spk_ui_accuracy, spk_ui_loss, exact_sr_accuracy, exact_sr_loss,
                    shared_unp_accuracy, shared_unp_loss, root_und_accuracy, root_und_loss]

    eval_op = [masked_lm_accuracy_op, masked_lm_loss_op, next_sentence_op, next_sentence_loss_op,
               spk_ui_accuracy_op, spk_ui_loss_op, exact_sr_accuracy_op, exact_sr_loss_op,
               shared_unp_accuracy_op, shared_unp_loss_op, root_und_accuracy_op, root_und_loss_op]

    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)])
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        for epoch in range(FLAGS.num_train_epochs):
            sess.run(iterator.initializer, feed_dict={filenames: [FLAGS.input_file]})
            run_epoch(epoch, sess, saver, FLAGS.output_dir, epoch_save_step, FLAGS.mid_save_step,
                      input_ids, eval_metrics, total_loss, train_op, eval_op)


if __name__ == "__main__":
    tf.compat.v1.app.run()
