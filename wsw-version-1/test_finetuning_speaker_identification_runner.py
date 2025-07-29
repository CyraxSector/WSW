import numpy as np
import pytest
import tensorflow as tf

from finetuning_speaker_identification_runner import create_model


class DummyBertConfig:
    hidden_size = 8
    hidden_act = "gelu"
    initializer_range = 0.02
    hidden_dropout_prob = 0.04
    attention_probs_dropout_prob = 0.04
    vocab_size = 50
    type_vocab_size = 2
    max_position_embeddings = 512
    num_hidden_layers = 6
    num_attention_heads = 8
    intermediate_size = 3072


def test_finetuning_speaker_identification_runner():
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    batch_size = 2
    seq_length = 10
    max_utr_num = 4
    hidden_size = 16
    num_labels = max_utr_num

    input_ids = tf.constant(np.random.randint(0, 100, size=(batch_size, seq_length)), dtype=tf.int32)
    input_mask = tf.constant(np.ones((batch_size, seq_length)), dtype=tf.int32)
    segment_ids = tf.constant(np.zeros((batch_size, seq_length)), dtype=tf.int32)
    speaker_ids = tf.constant(np.ones((batch_size, seq_length)), dtype=tf.int32)

    cls_positions = tf.constant(np.tile(np.arange(max_utr_num), (batch_size, 1)), dtype=tf.int32)
    rsp_position = tf.constant(np.full((batch_size, 1), max_utr_num), dtype=tf.int32)

    labels = tf.one_hot(np.random.randint(0, max_utr_num, size=(batch_size,)), depth=max_utr_num)
    labels = tf.cast(labels, tf.float32)
    config = DummyBertConfig()

    loss, logits, log_probs, accuracy = create_model(
        bert_config=config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        speaker_ids=speaker_ids,
        labels=labels,
        cls_positions=cls_positions,
        rsp_position=rsp_position,
        num_labels=labels,
        use_one_hot_embeddings=False
    )

    # Basic assertions
    assert loss.dtype == tf.float32
    assert logits.shape == (2, 4)
    assert log_probs.shape == (2, 4)
    assert accuracy.dtype == tf.float32
