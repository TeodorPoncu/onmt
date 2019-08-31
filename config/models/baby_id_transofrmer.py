import opennmt as onmt
import tensorflow as tf

def model():
    return onmt.models.Transformer(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key='source_words_vocabulary',
            embedding_size=512,
        ),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key='target_words_vocabulary',
            embedding_size=166,
        ),
        num_layers=4,
        num_units=128,
        num_heads=4,
        ffn_inner_dim=512
    )