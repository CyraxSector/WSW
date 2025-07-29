import numpy as np
import pytest
import tensorflow as tf

from finetuning_reply_utterance_selection_runner import create_model


class DummyBertModel:
    def __init__(self, config, is_training, input_ids, input_mask, token_type_ids, speaker_ids, use_one_hot_embeddings):
        self.output = tf.ones([input_ids.shape[0], config.hidden_size])

    def get_pooled_output(self):
        return self.output


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


@pytest.fixture(autouse=True)
def patch_modeling(monkeypatch):
    import types
    monkeypatch.setitem(__import__('sys').modules, 'modeling', types.SimpleNamespace(BertModel=DummyBertModel))


def test_reply_utterance_selection_create_model(patch_modeling):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    batch_size = 2
    seq_length = 5
    num_labels = 1

    input_ids = tf.constant(np.random.randint(0, 100, size=(batch_size, seq_length)), dtype=tf.int32)
    input_mask = tf.constant(np.ones((batch_size, seq_length)), dtype=tf.int32)
    segment_ids = tf.constant(np.zeros((batch_size, seq_length)), dtype=tf.int32)
    speaker_ids = tf.constant(np.zeros((batch_size, seq_length)), dtype=tf.int32)
    labels = tf.constant(np.random.randint(0, 2, size=(batch_size,)), dtype=tf.float32)
    ctx_id = tf.constant(np.zeros((batch_size,)), dtype=tf.int32)
    rsp_id = tf.constant(np.zeros((batch_size,)), dtype=tf.int32)
    config = DummyBertConfig()

    mean_loss, logits, probabilities, accuracy = create_model(
        bert_config=config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        speaker_ids=speaker_ids,
        labels=labels,
        ctx_id=ctx_id,
        rsp_id=rsp_id,
        num_labels=num_labels,
        use_one_hot_embeddings=False)

    # Assertions
    assert mean_loss.dtype == tf.float32
    assert logits.shape == (batch_size,)
    assert probabilities.shape == (batch_size, num_labels)
    assert accuracy.dtype == tf.float32
