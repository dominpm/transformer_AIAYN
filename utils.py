from custom_layers import * 
from transformer import Transformer
from dataset import tokenizer_en

def loss_function(real, pred, vocab_size=tokenizer_en.vocab_size(), label_smoothing=0.1):
    """
    real: (batch_size, seq_len)
    pred: (batch_size, seq_len, vocab_size) - logits
    """
    # Create a one-hot label with label smoothing
    real_one_hot = tf.one_hot(real, depth=vocab_size)  # shape: (batch_size, seq_len, vocab_size)
    real_smoothed = real_one_hot * (1.0 - label_smoothing) + (label_smoothing / vocab_size)

    # Mask padding tokens
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)  # (batch_size, seq_len)

    loss = tf.keras.losses.categorical_crossentropy(
        real_smoothed, pred, from_logits=True
    )  # shape: (batch_size, seq_len)

    loss *= mask  # zero-out padding

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


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


def sequence_accuracy_function(real, pred):
    """
    Computes sequence-level accuracy: the percentage of sequences where *all* tokens
    (excluding padding) are correctly predicted.
    
    Args:
        real: (batch_size, seq_len)
        pred: (batch_size, seq_len, vocab_size)
    Returns:
        Scalar float: percentage of completely correct sequences
    """
    pred_ids = tf.argmax(pred, axis=-1)  # (batch_size, seq_len)

    # Mask to ignore padding
    mask = tf.not_equal(real, 0)  # (batch_size, seq_len)

    # Equal where not padding
    correct_tokens = tf.equal(real, pred_ids)
    correct_tokens = tf.logical_and(mask, correct_tokens)

    # Count correct tokens per sequence
    correct_counts = tf.reduce_sum(tf.cast(correct_tokens, tf.int32), axis=-1)
    target_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)

    # Sequence is correct if all target tokens match
    fully_correct = tf.equal(correct_counts, target_lengths)

    # Return mean accuracy across the batch
    return tf.reduce_mean(tf.cast(fully_correct, tf.float32))


@tf.function(reduce_retracing=True)
def train_step(model, optimizer, inp, tar):
    tar_inp = tar[:, :-1]  # inputs to decoder
    tar_real = tar[:, 1:]  # expected outputs

    with tf.GradientTape() as tape:
        preds = model(inp, tar_inp, training=True)
        loss = loss_function(tar_real, preds)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    acc = accuracy_function(tar_real, preds)
    sqacc = sequence_accuracy_function(tar_real, preds)
    return loss, acc, sqacc


@tf.function(reduce_retracing=True)
def val_step(model, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    preds = model(inp, tar_inp, training=False)
    loss = loss_function(tar_real, preds)
    acc = accuracy_function(tar_real, preds)
    sqacc = sequence_accuracy_function(tar_real, preds)
    return loss, acc, sqacc

"""Finalmente, estamos en condiciones de crear un loop final para el entrenamiento, le meto earlystopping para evitar overfitting:"""

import os

def train(model : tf.keras.Model, optimizer, train_dataset, val_dataset,
          epochs, patience=3, min_delta=0.0, save_dir='artifacts/transformer/train_checkpoints'):
    """
    Entrena un Transformer con early-stopping,
    ahora con sequence-level accuracy incluido.
    """
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [], 'seq_acc': [], 'val_seq_acc': []}

    best_val_loss = float('inf')
    best_model_path = None
    wait = 0

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        # === ENTRENAMIENTO ==================================================
        total_loss = tf.keras.metrics.Mean()
        total_acc  = tf.keras.metrics.Mean()
        total_seq_acc = tf.keras.metrics.Mean()

        for batch, (inp, tar) in enumerate(train_dataset):
            loss, acc, seq_acc = train_step(model, optimizer, inp, tar)
            total_loss.update_state(loss)
            total_acc.update_state(acc)
            total_seq_acc.update_state(seq_acc)

            if (batch + 1) % 50 == 0:
                print(f'  Batch {batch + 1}: Loss={loss:.4f}  Acc={acc:.4f}  SeqAcc={seq_acc:.4f}')

        epoch_loss     = total_loss.result().numpy()
        epoch_accuracy = total_acc.result().numpy()
        epoch_seq_acc  = total_seq_acc.result().numpy()

        # === VALIDACIÓN =====================================================
        val_loss_metric = tf.keras.metrics.Mean()
        val_acc_metric  = tf.keras.metrics.Mean()
        val_seq_acc_metric = tf.keras.metrics.Mean()

        for inp_val, tar_val in val_dataset:
            loss, acc, seq_acc = val_step(model, inp_val, tar_val)
            val_loss_metric.update_state(loss)
            val_acc_metric.update_state(acc)
            val_seq_acc_metric.update_state(seq_acc)

        val_loss     = val_loss_metric.result().numpy()
        val_accuracy = val_acc_metric.result().numpy()
        val_seq_acc  = val_seq_acc_metric.result().numpy()

        # Guardamos en el history
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_accuracy)
        history['seq_acc'].append(epoch_seq_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_seq_acc'].append(val_seq_acc)

        print(f'  - Train   Loss={epoch_loss:.4f}  Acc={epoch_accuracy:.4f}  SeqAcc={epoch_seq_acc:.4f}')
        print(f'  - Val     Loss={val_loss:.4f}    Acc={val_accuracy:.4f}  SeqAcc={val_seq_acc:.4f}')

        # === GUARDADO DEL MODELO ============================================
        model_filename = f'model_epoch{epoch+1:02d}_valacc{val_accuracy:.4f}.keras'
        model_weights_filename = f'weights_model_epoch{epoch+1:02d}_valacc{val_accuracy:.4f}.weights.h5'
        model_path = os.path.join(save_dir, model_filename)
        model_path_weights = os.path.join(save_dir, "weights", model_weights_filename)
        model.save(model_path)
        model.save_weights(model_path_weights)
        print(f'  Model saved to {model_path}')
        print(f'  Model Weights saved to {model_path_weights}')

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
        try:
            print(f'\nRestoring best model from {best_model_path}...')
            restored_model = tf.keras.models.load_model(
                best_model_path,
                custom_objects={
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
            model.set_weights(restored_model.get_weights())
            print('Best model weights restored.')
        except:
            print("ERROR LOADING BEST MODEL CONTINUING")

    return history