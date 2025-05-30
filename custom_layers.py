import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from keras.saving import register_keras_serializable

@register_keras_serializable()
class EmbeddingLayer(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Inicializamos la matriz de embedding como un peso entrenable
        self.embeddings = self.add_weight(
            shape=(vocab_size, embedding_dim), # Vxd
            initializer='uniform',
            trainable=True,
            name="embedding_weights"
        )

    def call(self, x):
        # es lo mismo que return self.embeddings[x], pero tf nos da más eficiencia
        # en definitiva nos da las filas de E correspondientes a nuestro vector de tokens (su embedding)
        return tf.nn.embedding_lookup(self.embeddings, x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
        })
        return config

@register_keras_serializable()
class PositionalEncodingLayer(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(max_len, d_model)

    def _positional_encoding(self, max_len, d_model):
        # array de posiciones de 0 a max_len-1 (vertical)
        pos = np.arange(max_len)[:, np.newaxis]

        # array de 0 a d_model-1 (horizontal)
        i = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        # pares
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # impares
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...] # metemos el batch

        return tf.cast(pos_encoding, dtype=tf.float32) # sino tf se queja

    def call(self, x):
        seq_len = tf.shape(x)[1] # sequence length

        # Sumamos el embedding con su codificación posicional
        # Cortamos pos_encoding para que coincida si la secuencia es más corta que max_len
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config
    

@register_keras_serializable()
class ScaledDotProductAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, q, k, v, mask=None):
        """
        q: consultas -> shape (batch_size, T_q, d_k)
        k: claves -> shape (batch_size, T_k, d_k)
        v: valores -> shape (batch_size, T_k, d_v)
        mask: máscara opcional -> shape (batch_size, T_q, T_k)
        """

        # Producto punto entre consultas y claves transpuestas
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, T_q, T_k)

        # Escalamos por la raíz cuadrada de la dimensión de las claves
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

        # Aplicamos la máscara si existe (añadiendo un gran valor negativo donde la máscara sea 0)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax en el eje de las claves para obtener pesos de atención
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, T_q, T_k)

        # Multiplicamos los pesos de atención por los valores
        output = tf.matmul(attention_weights, v)  # (batch_size, T_q, d_v)

        return output, attention_weights

    def get_config(self):
        config = super().get_config()
        return config
    

@register_keras_serializable()
class MultiHeadAttentionLayer(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model debe ser divisible entre num_heads"

        # Dimensión de cada cabeza
        self.depth = d_model // num_heads

        # Proyecciones lineales para Q, K y V
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        # Proyección final tras la concatenación
        self.wo = layers.Dense(d_model)

        # Instancia del mecanismo de atención escalada
        self.attention = ScaledDotProductAttentionLayer()

    def split_heads(self, x, batch_size):
        """
        Divide la dimensión d_model en múltiples cabezas.
        Entrada: (batch_size, seq_len, d_model)
        Salida: (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # Este cambio de ejes reordena el tensor para dejar las cabezas como segundo eje:

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        # Proyecciones lineales
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        # Dividir en múltiples cabezas
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Aplicar atención escalada por producto punto en cada cabeza
        scaled_attention, attention_weights = self.attention(q, k, v, mask) # Usamos los batches para poder calcularlos todos a la vez

        # Unir las cabezas, primero transponemos y luego reestructuramos
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # Proyección final
        output = self.wo(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config
    
    
@register_keras_serializable()
class LayerNormalization(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # escalado y desplazamiento aprendibles
        self.gamma = self.add_weight(
            name="gamma",
            shape=input_shape[-1:],
            initializer="ones",
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[-1:],
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        # Calculamos media y varianza sobre la última dimensión (d_model)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)

        x_norm = (x - mean) / tf.sqrt(variance + self.epsilon)

        # Aplicamos reescalado y desplazamiento
        return self.gamma * x_norm + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
        })
        return config
    

@register_keras_serializable()
class PositionwiseFeedForwardLayer(layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.dense1 = layers.Dense(d_ff, activation='relu')  # Primera proyección con ReLU
        self.dense2 = layers.Dense(d_model) # Segunda proyección sin activación
        self.dropout = layers.Dropout(dropout_rate) # para evitar overfitting

    def call(self, x, training=False):
        x = self.dense1(x) # (batch_size, seq_len, d_ff)
        x = self.dropout(x, training=training)
        x = self.dense2(x) # (batch_size, seq_len, d_model)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        })
        return config
    

def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)  # 1 donde haya padding
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Decoder padding mask (para encoder-decoder attention)
    dec_padding_mask = create_padding_mask(inp)

    # Look-ahead mask para el target
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])

    # Padding mask sobre el target (para ignorar padding en el decoder también)
    tar_padding_mask = create_padding_mask(tar)

    # Combined mask para el primer bloque del decoder, abajo lo vemos
    combined_mask = tf.maximum(look_ahead_mask, tar_padding_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

@register_keras_serializable()
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttentionLayer(d_model, num_heads)               # Multi-Head Attention
        self.ffn = PositionwiseFeedForwardLayer(d_model, d_ff, dropout_rate)  # Feed-Forward Network

        self.layernorm1 = LayerNormalization()  # Primera LayerNorm (tras atención)
        self.layernorm2 = LayerNormalization()  # Segunda LayerNorm (tras FFN)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        # 1. Multi-Head Attention
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)

        # 2. Residual connection + LayerNorm
        out1 = self.layernorm1(x + attn_output)

        # 3. Feed-Forward Network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)

        # 4. Residual connection + LayerNorm
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        })
        return config
    
@register_keras_serializable()
class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Primer bloque: Masked Multi-Head Self-Attention
        self.mha1 = MultiHeadAttentionLayer(d_model, num_heads)

        # Segundo bloque: Multi-Head Encoder-Decoder Attention
        self.mha2 = MultiHeadAttentionLayer(d_model, num_heads)

        # Feed-Forward Network
        self.ffn = PositionwiseFeedForwardLayer(d_model, d_ff, dropout_rate)

        # LayerNorms
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()

        # Dropouts
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        """
        x: (batch_size, target_seq_len, d_model) -> entrada del decoder
        enc_output: (batch_size, input_seq_len, d_model) -> salida del encoder
        look_ahead_mask: máscara look-ahead combinada con padding
        padding_mask: máscara de padding para el encoder-decoder attention
        """

        # 1. Masked Multi-Head Self-Attention
        attn1, _ = self.mha1(x, x, x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 2. Encoder-Decoder Multi-Head Attention
        attn2, _ = self.mha2(enc_output, enc_output, out1, mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # 3. Feed-Forward Network
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        })
        return config