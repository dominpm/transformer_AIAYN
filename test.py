# import matplotlib.pyplot as plt
# import pickle

# with open("/home/asidpm/transformer_dpm/transformer_AIAYN/artifacts/transformer/history.pkl", "rb") as f:
#     hist = pickle.load(f)

# plt.plot(hist["loss"], label="Train Loss")
# plt.plot(hist["val_loss"], label="Val Loss")
# plt.plot(hist["acc"], label="Train Acc")
# plt.plot(hist["val_acc"], label="Val Acc")
# # plt.plot(hist["seq_acc"], label="Train SeqAcc")
# # plt.plot(hist["val_seq_acc"], label="Val SeqAcc")
# plt.legend()
# plt.title("Training History")
# plt.savefig("training_curves.png")

import tensorflow as tf
from tqdm import tqdm
from dataset import tokenizer_aztr, tokenizer_en, MAX_LENGTH
from utils import sequence_accuracy_function, accuracy_function
from transformer import Transformer
from main import params_config
from dataset import tst_examples  # asegúrate de tenerlo cargado como ya mencionaste


# --------- TOKENIZACIÓN Y PREPARACIÓN ---------
def encode(aztr_text, en_text):
    aztr_ids = tokenizer_aztr.encode(aztr_text.numpy().decode('utf-8'), out_type=int)
    en_ids = tokenizer_en.encode(en_text.numpy().decode('utf-8'), out_type=int)
    aztr_ids = [tokenizer_aztr.bos_id()] + aztr_ids + [tokenizer_aztr.eos_id()]
    en_ids = [tokenizer_en.bos_id()] + en_ids + [tokenizer_en.eos_id()]
    return aztr_ids, en_ids

def tf_encode(aztr_text, en_text):
    result_aztr, result_en = tf.py_function(encode, [aztr_text, en_text], [tf.int64, tf.int64])
    result_aztr.set_shape([None])
    result_en.set_shape([None])
    return result_aztr, result_en

def filter_max_length(aztr, en, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(aztr) <= max_length,
                          tf.size(en) <= max_length)

# Preparar el conjunto de test
BATCH_SIZE = 128
test_dataset = tst_examples.map(tf_encode)
test_dataset = test_dataset.filter(filter_max_length)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=([MAX_LENGTH], [MAX_LENGTH]))
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# --------- CARGAR EL MODELO ---------
model = Transformer(**params_config)
dummy_inp = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
dummy_tar = tf.zeros((1, MAX_LENGTH + 1), dtype=tf.int32)
model(dummy_inp, dummy_tar, training=False)

model_path = 'artifacts/transformer/train_checkpoints/weights/weights_model_epoch23_valacc0.4206.weights.h5'
model.load_weights(model_path)
print(f"✅ Modelo cargado desde: {model_path}")

# --------- EVALUACIÓN ---------
token_acc_metric = tf.keras.metrics.Mean()
seq_acc_metric = tf.keras.metrics.Mean()

for inp, tar in tqdm(test_dataset, desc="Evaluando test set"):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    pred = model(inp, tar_inp, training=False)
    token_acc = accuracy_function(tar_real, pred)
    seq_acc = sequence_accuracy_function(tar_real, pred)

    token_acc_metric.update_state(token_acc)
    seq_acc_metric.update_state(seq_acc)

print("\n📊 Resultados en el conjunto de test:")
print(f"🔹 Token Accuracy    : {token_acc_metric.result().numpy():.4f}")
print(f"🔹 Sequence Accuracy : {seq_acc_metric.result().numpy():.4f}")