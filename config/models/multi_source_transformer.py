"""Define the Google's Transformer model."""

import tensorflow as tf
import opennmt as onmt
import math

from opennmt.models.sequence_to_sequence import SequenceToSequence, EmbeddingsSharingLevel
from opennmt.encoders.encoder import ParallelEncoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers.position import PositionEmbedder, PositionEncoder
from opennmt.utils.misc import merge_dict

class BlindPosition(PositionEncoder):

  def encode(self, positions, depth, dtype=tf.float32):
    batch_size = tf.shape(positions)[0]
    positions = tf.cast(positions, tf.float32)

    log_timescale_increment = math.log(1) / (depth / 2 - 1)
    inv_timescales = tf.exp(tf.range(depth / 2, dtype=tf.float32) * -log_timescale_increment)
    inv_timescales = tf.reshape(tf.tile(inv_timescales, [batch_size]), [batch_size, -1])
    scaled_time = tf.expand_dims(positions, -1) * tf.expand_dims(inv_timescales, 1)
    encoding = tf.concat([scaled_time, scaled_time], axis=2)
    return tf.cast(encoding, dtype)


class CustomTransformer(SequenceToSequence):
  """Attention-based sequence-to-sequence model as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               source_inputter,
               target_inputter,
               num_layers,
               num_units,
               num_heads,
               ffn_inner_dim,
               dropout=0.1,
               attention_dropout=0.1,
               relu_dropout=0.1,
               position_encoder=SinusoidalPositionEncoder(),
               decoder_self_attention_type="scaled_dot",
               share_embeddings=EmbeddingsSharingLevel.NONE,
               share_encoders=False,
               alignment_file_key="train_alignments",
               name="transformer"):
    """Initializes a Transformer model.

    Args:
      source_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process
        the source data. If this inputter returns parallel inputs, a multi
        source Transformer architecture will be constructed.
      target_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process
        the target data. Currently, only the
        :class:`opennmt.inputters.text_inputter.WordEmbedder` is supported.
      num_layers: The shared number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in each self-attention layers.
      ffn_inner_dim: The inner dimension of the feed forward layers.
      dropout: The probability to drop units in each layer output.
      attention_dropout: The probability to drop units from the attention.
      relu_dropout: The probability to drop units from the ReLU activation in
        the feed forward layer.
      position_encoder: A :class:`opennmt.layers.position.PositionEncoder` to
        apply on the inputs.
      decoder_self_attention_type: Type of self attention in the decoder,
        "scaled_dot" or "average" (case insensitive).
      share_embeddings: Level of embeddings sharing, see
        :class:`opennmt.models.sequence_to_sequence.EmbeddingsSharingLevel`
        for possible values.
      share_encoders: In case of multi source architecture, whether to share the
        separate encoders parameters or not.
      alignment_file_key: The data configuration key of the training alignment
        file to support guided alignment.
      name: The name of this model.
    """
    encoders = [
        SelfAttentionEncoder(
            num_layers,
            num_units=num_units,
            num_heads=num_heads,
            ffn_inner_dim=ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            position_encoder=BlindPosition())
         for _ in range(2)]
    print("Len is " + str(len(encoders)))
    encoders += ([
        SelfAttentionEncoder(
            num_layers,
            num_units=num_units,
            num_heads=num_heads,
            ffn_inner_dim=ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            position_encoder=position_encoder)
        for _ in range(2)]
    )
    print("Len is " + str(len(encoders)))
    if len(encoders) > 1:
      encoder = ParallelEncoder(
          encoders,
          outputs_reducer=None,
          states_reducer=None,
          share_parameters=share_encoders)
    else:
      encoder = encoders[0]
    decoder = SelfAttentionDecoder(
        num_layers,
        num_units=num_units,
        num_heads=num_heads,
        ffn_inner_dim=ffn_inner_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        relu_dropout=relu_dropout,
        position_encoder=position_encoder,
        self_attention_type=decoder_self_attention_type)

    self._num_units = num_units
    super(CustomTransformer, self).__init__(
        source_inputter,
        target_inputter,
        encoder,
        decoder,
        share_embeddings=share_embeddings,
        alignment_file_key=alignment_file_key,
        daisy_chain_variables=True,
        name=name)

  def auto_config(self, num_devices=1):
    config = super(CustomTransformer, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "average_loss_in_time": True,
            "label_smoothing": 0.1,
            "optimizer": "LazyAdamOptimizer",
            "optimizer_params": {
                "beta1": 0.9,
                "beta2": 0.998
            },
            "learning_rate": 2.0,
            "decay_type": "noam_decay_v2",
            "decay_params": {
                "model_dim": self._num_units,
                "warmup_steps": 8000
            }
        },
        "train": {
            "effective_batch_size": 2048,
            "batch_size": 1024,
            "batch_type": "tokens",
            "maximum_features_length": 100,
            "maximum_labels_length": 100,
            "keep_checkpoint_max": 8,
            "average_last_checkpoints": 8
        }
    })

  def _initializer(self, params):
    return tf.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", dtype=self.dtype)

class MultiSourceTransformer(CustomTransformer):

  def __init__(self):
    super(MultiSourceTransformer, self).__init__(
      source_inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_1",
              embedding_size=512),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_2",
              embedding_size=512),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_3",
              embedding_size=512),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_4",
              embedding_size=512)]),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_vocabulary",
          embedding_size=256),
      num_layers=4,
      num_units=256,
      num_heads=6,
      ffn_inner_dim=512,
      dropout=0.1,
      attention_dropout=0.1,
      relu_dropout=0.1,
      share_encoders=False)

  def auto_config(self, num_devices=1):
    config = super(MultiSourceTransformer, self).auto_config(num_devices=num_devices)
    max_length = config["train"]["maximum_features_length"]
    return misc.merge_dict(config, {
        "train": {
            "maximum_features_length": [max_length, max_length]
        }
    })


model = MultiSourceTransformer
