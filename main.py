import pickle
from pathlib import Path
from custom_layers import *
from dataset import *
from utils import *
from transformer import *

params_config = {
    "num_encoder_layers":10,
    "num_decoder_layers":10,
    'd_model': 128,
    'num_heads': 8,
    'd_ff': 512,
    'input_vocab_size': tokenizer_aztr.vocab_size(),
    'target_vocab_size': tokenizer_en.vocab_size(),
    'max_positional_encoding_input': MAX_LENGTH,
    'max_positional_encoding_target': MAX_LENGTH,
    'dropout_rate': 0.1,
}

# Instanciación del Transformer
automated_translation_transformer = Transformer(**params_config)

# Scheduler para warm-up
learning_rate = NoamSchedule(params_config['d_model'])

# Optimizer Adam ( con valores del paper )
optimizer = tf.keras.optimizers.Adam(
    learning_rate,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9
)

# hago un poco de "trampa" para que el modelo llame a build de todas las capas y ver un buen summary
dummy_inp = tf.random.uniform((1, MAX_LENGTH), dtype=tf.int32, minval=0, maxval=params_config['input_vocab_size'])
dummy_tar = tf.random.uniform((1, MAX_LENGTH+1), dtype=tf.int32, minval=0, maxval=params_config['target_vocab_size'])
automated_translation_transformer(dummy_inp, dummy_tar, training=False)

print("Model summary : ")
print(automated_translation_transformer.summary())

# ---------- warm-up ----------
inp, tar = next(iter(train_dataset))
_ = train_step(automated_translation_transformer,
               optimizer,
               inp,
               tar)

history = train(
    model=automated_translation_transformer,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=50,
    patience=5,
    min_delta=0.001
)


out_dir = Path("artifacts/transformer")
out_dir.mkdir(exist_ok=True)


hist_path = out_dir / "history.pkl"
with hist_path.open("wb") as f:
    pickle.dump(history, f)

print("Historial guardado en:", hist_path.resolve())

model_path = out_dir / "transformer_savedmodel_last.keras"
automated_translation_transformer.save(model_path)
print("Modelo guardado en:", model_path.resolve())