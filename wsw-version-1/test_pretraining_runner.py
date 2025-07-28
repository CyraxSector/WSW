from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

import modeling_speaker as modeling
from pretraining_runner import (get_speaker_utterance_identification_output, get_exact_speaker_recognition_output,
                                get_root_utterance_node_detection_output)


class DummyBertConfig:
    hidden_size = 8
    hidden_act = "gelu"
    initializer_range = 0.02


def dummy_gather_indexes(sequence_tensor, positions):
    shape = tf.shape(positions)
    batch_size = shape[0]
    seq_len = shape[1]
    dim = tf.shape(sequence_tensor)[-1]
    return tf.ones((batch_size * seq_len, dim), dtype=tf.float32)


def dummy_layer_norm(x):
    return x


@pytest.fixture
def dummy_inputs():
    batch_size = 2
    seq_length = 6
    max_utr_num = 3
    hidden_size = 8
    max_predictions = 2

    input_tensor = tf.constant(np.random.rand(batch_size, seq_length, hidden_size), dtype=tf.float32)
    cls_positions = tf.constant([[0, 2, 4], [1, 3, 5]], dtype=tf.int32)  # shape [2, 3]
    masked_sr_positions = tf.constant([[0], [1]], dtype=tf.int32)  # shape [2, 1]
    masked_sr_labels = tf.constant([[1], [2]], dtype=tf.int32)
    masked_sr_weights = tf.constant([[1.0], [1.0]], dtype=tf.float32)
    root_labels = tf.constant(np.random.randint(0, 2, size=(batch_size,)), dtype=tf.int32)

    return (DummyBertConfig(), input_tensor, cls_positions, masked_sr_positions, masked_sr_labels, masked_sr_weights,
            root_labels)


@pytest.mark.parametrize("batch_size,max_utr_num,width,max_predictions_per_seq_ar", [
    (2, 4, 16, 2),
])
def test_get_speaker_utterance_identification_output(batch_size, max_utr_num, width, max_predictions_per_seq_ar):
    tf.compat.v1.reset_default_graph()
    modeling.get_shape_list = lambda x, expected_rank=None: x.shape.as_list()
    modeling.get_activation = lambda name: tf.nn.relu
    modeling.create_initializer = lambda range_: tf.keras.initializers.TruncatedNormal(stddev=range_)
    modeling.layer_norm = dummy_layer_norm

    global gather_indexes
    gather_indexes = dummy_gather_indexes

    bert_config = SimpleNamespace(
        hidden_size=width,
        hidden_act="relu",
        initializer_range=0.02
    )

    input_tensor = tf.constant(np.random.rand(batch_size, 128, width), dtype=tf.float32)
    cls_positions = tf.constant(np.zeros((batch_size, max_utr_num), dtype=np.int32))
    adr_recog_positions = tf.constant(np.zeros((batch_size, max_predictions_per_seq_ar), dtype=np.int32))
    adr_recog_labels = tf.constant(np.zeros((batch_size, max_predictions_per_seq_ar), dtype=np.int32))
    adr_recog_weights = tf.constant(np.ones((batch_size, max_predictions_per_seq_ar), dtype=np.float32))

    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)])
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        loss, per_example_loss, log_probs = get_speaker_utterance_identification_output(
            bert_config,
            input_tensor,
            cls_positions,
            adr_recog_positions,
            adr_recog_labels,
            adr_recog_weights
        )
        sess.run(tf.compat.v1.global_variables_initializer())
        loss_val, per_example_loss_val, log_probs_val = sess.run([loss, per_example_loss, log_probs])

    assert isinstance(loss_val, float)
    assert per_example_loss_val.shape[0] == batch_size * max_predictions_per_seq_ar
    assert log_probs_val.shape[1] == max_utr_num


def test_get_exact_speaker_recognition_output(dummy_inputs):
    bert_config, input_tensor, cls_positions, masked_sr_positions, masked_sr_labels, masked_sr_weights = dummy_inputs

    tf.compat.v1.reset_default_graph()
    tf.config.experimental.set_virtual_device_configuration(tf.config.experimental.list_physical_devices('GPU')[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)])
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        loss, per_example_loss, log_probs = get_exact_speaker_recognition_output(
            bert_config,
            input_tensor,
            cls_positions,
            masked_sr_positions,
            masked_sr_labels,
            masked_sr_weights
        )
        sess.run(tf.compat.v1.global_variables_initializer())
        loss_val, pel_val, logp_val = sess.run([loss, per_example_loss, log_probs])

        assert loss_val > 0
        assert pel_val.shape[0] == 2
        assert logp_val.shape == (2, 3)


def test_get_root_utterance_node_detection_output(dummy_inputs):
    bert_config, input_tensor, root_labels = dummy_inputs

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:

        loss, per_example_loss, log_probs = get_root_utterance_node_detection_output(
            bert_config, input_tensor, root_labels
        )

        sess.run(tf.compat.v1.global_variables_initializer())
        loss_val, pel_val, logp_val = sess.run([loss, per_example_loss, log_probs])

        # Assertions
        assert isinstance(loss_val, np.floating)
        assert pel_val.shape == (3,)
        assert logp_val.shape == (3, 2)
        assert np.allclose(np.exp(logp_val).sum(axis=1), 1.0, atol=1e-5)
