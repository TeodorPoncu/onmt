import tensorflow as tf
import opennmt as onmt

def model():
    return onmt.models.SequenceToSequence(
        source_inputter=
            onmt.inputters.ParallelInputter([
                onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_1",
              embedding_size=256),
            onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_2",
              embedding_size=256),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_3",
              embedding_size=256),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_4",
              embedding_size=256)]),
        target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_vocabulary",
          embedding_size=256),
        encoder=onmt.encoders.ParallelEncoder([
            onmt.encoders.SequentialEncoder([
                onmt.encoders.SelfAttentionEncoder(
                    num_layers=4,
                    num_units=256,
                    position_encoder=onmt.layers.PositionEmbedder()
                ),
                onmt.encoders.SelfAttentionEncoder(
                    num_layers=4,
                    num_units=256,
                    position_encoder=onmt.layers.PositionEmbedder(),
                )],
                states_reducer=onmt.layers.MultiplyReducer()
            ),
            onmt.encoders.SelfAttentionEncoder(
                num_layers=4,
                num_units=256,
                position_encoder=onmt.layers.SinusoidalPositionEncoder()
            ),
            onmt.encoders.SelfAttentionEncoder(
                num_layers=4,
                num_units=256,
                position_encoder=onmt.layers.SinusoidalPositionEncoder()
            )],
            outputs_reducer=onmt.layers.ConcatReducer()),
        decoder=onmt.decoders.SelfAttentionDecoder(
            num_layers=8,
            num_units=712
        ))