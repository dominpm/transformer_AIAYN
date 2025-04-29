from custom_layers import * 
from transformer import Transformer

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    # Calcula el loss por token
    loss_ = loss_object(real, pred)  # (batch_size, seq_len)

    # Crear máscara para ignorar padding tokens (token == 0)
    mask = tf.cast(tf.not_equal(real, 0), dtype=loss_.dtype)

    # Aplicar la máscara: loss solo donde mask == 1
    loss_ *= mask

    # Devolver loss promedio por tokens válidos ( no padding )
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        arg1 = tf.math.rsqrt(step)  # 1/sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)  # step * (1/warmup_steps)^1.5

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model.numpy(),
            "warmup_steps": self.warmup_steps,
        })
        return config

"""Definamos ahora un accuracy, para ir viendo en el entrenamiento cómo vamos."""

def accuracy_function(real, pred):
    """
    real: (batch_size, seq_len) -> tokens reales
    pred: (batch_size, seq_len, vocab_size) -> predicciones (logits)
    """
    # Predicción final: tomar el índice con mayor probabilidad
    pred = tf.argmax(pred, axis=-1)  # (batch_size, seq_len)

    # Comparar tokens
    accuracies = tf.equal(real, pred)  # (batch_size, seq_len)

    # Crear máscara para ignorar padding
    mask = tf.math.not_equal(real, 0)  # (batch_size, seq_len)
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # Devuelve el porcentaje de tokens correctos entre tokens válidos
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

"""Podríamos entrenarlo de forma nativa, pero mejor metámonos a entender como se ve un training step ( más divertido que pegarle al `.fit` ), defino la siguiente función:"""

@tf.function(reduce_retracing=True)
def train_step(model, optimizer, inp, tar):
    with tf.GradientTape() as tape:
        preds, real = model(inp, tar, training=True)
        loss = loss_function(real, preds)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    acc = accuracy_function(real, preds)
    return loss, acc


@tf.function(reduce_retracing=True)
def val_step(model, inp, tar):
    preds, real = model(inp, tar, training=False)
    loss = loss_function(real, preds)
    acc  = accuracy_function(real, preds)
    return loss, acc

"""Finalmente, estamos en condiciones de crear un loop final para el entrenamiento, le meto earlystopping para evitar overfitting:"""

import os

def train(model, optimizer, train_dataset, val_dataset,
          epochs, patience=3, min_delta=0.0, save_dir='artifacts/transformer/train_checkpoints'):
    """
    Entrena un Transformer con early-stopping
    y guarda el modelo tras cada epoch con su val_accuracy en el nombre.
    Al terminar, restaura el modelo al mejor estado (menor val_loss).
    """
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    best_val_loss = float('inf')
    best_model_path = None
    wait = 0

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        # === ENTRENAMIENTO ==================================================
        total_loss = tf.keras.metrics.Mean()
        total_acc  = tf.keras.metrics.Mean()

        for batch, (inp, tar) in enumerate(train_dataset):
            loss, acc = train_step(model, optimizer, inp, tar)
            total_loss.update_state(loss)
            total_acc.update_state(acc)

            if (batch + 1) % 50 == 0:
                print(f'  Batch {batch + 1}: Loss={loss:.4f}  Acc={acc:.4f}')

        epoch_loss     = total_loss.result().numpy()
        epoch_accuracy = total_acc.result().numpy()

        # === VALIDACIÓN =====================================================
        val_loss_metric = tf.keras.metrics.Mean()
        val_acc_metric  = tf.keras.metrics.Mean()

        for inp_val, tar_val in val_dataset:
            loss, acc = val_step(model, inp_val, tar_val)
            val_loss_metric.update_state(loss)
            val_acc_metric.update_state(acc)

        val_loss     = val_loss_metric.result().numpy()
        val_accuracy = val_acc_metric.result().numpy()

        # Guardamos en el history
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)

        print(f'  - Train   Loss={epoch_loss:.4f}  Acc={epoch_accuracy:.4f}')
        print(f'  - Val     Loss={val_loss:.4f}    Acc={val_accuracy:.4f}')

        # === GUARDADO DEL MODELO ============================================
        model_filename = f'model_epoch{epoch+1:02d}_valacc{val_accuracy:.4f}.keras'
        model_path = os.path.join(save_dir, model_filename)
        model.save(model_path)
        print(f'  Model saved to {model_path}')

        # Guardar mejor modelo
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            best_model_path = model_path
            wait = 0
            print(f'  New best val_loss ({best_val_loss:.4f}) – patience reset')
        else:
            wait += 1
            print(f'  No val improvement → patience {wait}/{patience}')
            if wait >= patience:
                print('\nEarly-stopping: validation did not improve')
                break

    # === RESTORE BEST MODEL =================================================
    if best_model_path is not None:
        print(f'\nRestoring best model from {best_model_path}...')
        restored_model = tf.keras.models.load_model(
            best_model_path,
            custom_objects={  # Si usas clases personalizadas
                "EmbeddingLayer": EmbeddingLayer,
                "PositionalEncodingLayer": PositionalEncodingLayer,
                "ScaledDotProductAttentionLayer": ScaledDotProductAttentionLayer,
                "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
                "LayerNormalization": LayerNormalization,
                "PositionwiseFeedForwardLayer": PositionwiseFeedForwardLayer,
                "TransformerEncoderLayer": TransformerEncoderLayer,
                "TransformerDecoderLayer": TransformerDecoderLayer,
                "Transformer": Transformer,
                "NoamSchedule": NoamSchedule
            }
        )
        # Copiamos pesos restaurados al modelo original
        model.set_weights(restored_model.get_weights())
        print('Best model weights restored.')

    return history