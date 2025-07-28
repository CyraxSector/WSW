from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

import modeling_speaker as modeling
from pretraining_runner import get_speaker_utterance_identification_output


def dummy_gather_indexes(sequence_tensor, positions):
    shape = tf.shape(positions)
    batch_size = shape[0]
    seq_len = shape[1]
    dim = tf.shape(sequence_tensor)[-1]
    return tf.ones((batch_size * seq_len, dim), dtype=tf.float32)


def dummy_layer_norm(x):
    return x


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

    with tf.compat.v1.Session() as sess:
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
