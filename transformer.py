from custom_layers import *
from keras.saving import register_keras_serializable

@register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff,
                 input_vocab_size, target_vocab_size, max_positional_encoding_input,
                 max_positional_encoding_target, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_positional_encoding_input = max_positional_encoding_input
        self.max_positional_encoding_target = max_positional_encoding_target
        self.dropout_rate = dropout_rate

        # Embeddings ( no tienen por que ser iguales )
        self.encoder_embedding = EmbeddingLayer(input_vocab_size, d_model)
        self.decoder_embedding = EmbeddingLayer(target_vocab_size, d_model)

        # Positional Encodings (no entrenables)
        self.positional_encoding_encoder = PositionalEncodingLayer(max_positional_encoding_input, d_model)
        self.positional_encoding_decoder = PositionalEncodingLayer(max_positional_encoding_target, d_model)

        # Stack de Encoder Layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_encoder_layers)
        ]

        # Stack de Decoder Layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_decoder_layers)
        ]

        # Capa final: Dense para predecir vocabulario target
        self.final_layer = layers.Dense(target_vocab_size)

        self.dropout_enc = layers.Dropout(dropout_rate)
        self.dropout_dec = layers.Dropout(dropout_rate)

    def call(self, inp, tar, training=False):
        """
        inp: (batch_size, input_seq_len)
        tar: (batch_size, target_seq_len) → should be tar_inp externally
        """
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)

        # Encoder
        enc_output = self.encoder_embedding(inp)
        enc_output = self.positional_encoding_encoder(enc_output)
        enc_output = self.dropout_enc(enc_output, training=training)

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, training=training, mask=enc_padding_mask)

        # Decoder
        dec_output = self.decoder_embedding(tar)
        dec_output = self.positional_encoding_decoder(dec_output)
        dec_output = self.dropout_dec(dec_output, training=training)

        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, training=training,
                                look_ahead_mask=combined_mask, padding_mask=dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, vocab_size)
        return final_output  


    # def build(self, input_shape):
    #     encoder_input_shape, decoder_input_shape = input_shape

    #     self.encoder_embedding.build(encoder_input_shape)
    #     self.decoder_embedding.build(decoder_input_shape)

    #     self.positional_encoding_encoder.build(encoder_input_shape)
    #     self.positional_encoding_decoder.build(decoder_input_shape)

    #     for layer in self.encoder_layers:
    #         layer.build(encoder_input_shape)

    #     for layer in self.decoder_layers:
    #         layer.build([decoder_input_shape, encoder_input_shape])

    #     self.final_layer.build((decoder_input_shape[0], decoder_input_shape[1], self.d_model))

    #     self.dropout_enc.build(encoder_input_shape)
    #     self.dropout_dec.build(decoder_input_shape)

    #     super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "input_vocab_size": self.input_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "max_positional_encoding_input": self.max_positional_encoding_input,
            "max_positional_encoding_target": self.max_positional_encoding_target,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)