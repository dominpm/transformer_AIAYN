import tensorflow_datasets as tfds
import sentencepiece as spm
import tensorflow as tf

examples, metadata = tfds.load('ted_hrlr_translate/aztr_to_en', with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

artifacts_folder_cat = "./artifacts/corpus_and_tokenizers/"
tokenizer_aztr = spm.SentencePieceProcessor(model_file=artifacts_folder_cat+'aztr.model')
tokenizer_en = spm.SentencePieceProcessor(model_file=artifacts_folder_cat+'en.model')

BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 100

def encode(aztr_text, en_text):
    aztr_ids = tokenizer_aztr.encode(aztr_text.numpy().decode('utf-8'), out_type=int)
    en_ids = tokenizer_en.encode(en_text.numpy().decode('utf-8'), out_type=int)
    return aztr_ids, en_ids

def tf_encode(aztr_text, en_text):
    result_aztr, result_en = tf.py_function(encode, [aztr_text, en_text], [tf.int64, tf.int64])
    result_aztr.set_shape([None])
    result_en.set_shape([None])
    return result_aztr, result_en

def filter_max_length(aztr, en, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(aztr) <= max_length,
                           tf.size(en) <= max_length)

# preparo los dataset
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([MAX_LENGTH], [MAX_LENGTH]))
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length)
val_dataset = val_dataset.padded_batch(BATCH_SIZE, padded_shapes=([MAX_LENGTH], [MAX_LENGTH]))