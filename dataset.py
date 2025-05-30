import tensorflow_datasets as tfds
import sentencepiece as spm
import tensorflow as tf

examples, metadata = tfds.load('ted_hrlr_translate/aztr_to_en', with_info=True, as_supervised=True)

train_examples, val_examples, tst_examples = examples['train'], examples['validation'], examples['test']

# train_examples, val_examples = train_examples.take(100), val_examples.take(100) # just to test this
artifacts_folder_cat = "./artifacts/corpus_and_tokenizers/"
tokenizer_aztr = spm.SentencePieceProcessor(model_file=artifacts_folder_cat+'aztr.model')
tokenizer_en = spm.SentencePieceProcessor(model_file=artifacts_folder_cat+'en.model')

BUFFER_SIZE = 20000
BATCH_SIZE = 128
MAX_LENGTH = 256

print("BOS ID:", tokenizer_en.bos_id())
print("EOS ID:", tokenizer_en.eos_id())
print("PAD ID:", tokenizer_en.pad_id())
print("UNK ID:", tokenizer_en.unk_id())  

print("BOS token:", tokenizer_en.id_to_piece(tokenizer_en.bos_id()))
print("EOS token:", tokenizer_en.id_to_piece(tokenizer_en.eos_id()))
print("PAD token:", tokenizer_en.id_to_piece(tokenizer_en.pad_id()))
print("UNK token:", tokenizer_en.id_to_piece(tokenizer_en.unk_id()))

# import numpy as np

# az_lengths = []
# en_lengths = []

# # Analyze only training set
# for aztr, en in tfds.as_numpy(train_examples):
#     aztr_tokens = [tokenizer_aztr.bos_id()] + tokenizer_aztr.encode(aztr.decode('utf-8')) + [tokenizer_aztr.eos_id()]
#     en_tokens = [tokenizer_en.bos_id()] + tokenizer_en.encode(en.decode('utf-8')) + [tokenizer_en.eos_id()]
    
#     az_lengths.append(len(aztr_tokens))
#     en_lengths.append(len(en_tokens))

# # Convert to numpy for stats
# az_lengths = np.array(az_lengths)
# en_lengths = np.array(en_lengths)

# print("\n📊 Tokenized Sequence Length Stats (with BOS/EOS):")
# print(f"🔹 Azerí → Max: {az_lengths.max()},  Mean: {az_lengths.mean():.2f},  95%: {np.percentile(az_lengths, 95)},  99%: {np.percentile(az_lengths, 99)}")
# print(f"🔹 English → Max: {en_lengths.max()},  Mean: {en_lengths.mean():.2f},  95%: {np.percentile(en_lengths, 95)},  99%: {np.percentile(en_lengths, 99)}")


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

# preparo los dataset
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([MAX_LENGTH], [MAX_LENGTH]))
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

for example in train_dataset.take(1):
    azbatch = example[0]
    enbatch = example[1]
    az_sentence = azbatch[0].numpy().tolist()
    en_sentence = enbatch[0].numpy().tolist()

    # Remove padding tokens if necessary (optional)
    az_sentence = [id for id in az_sentence ]
    en_sentence = [id for id in en_sentence ]

    print("AZBATCH DECODED : ", [tokenizer_aztr.id_to_piece(id) for id in az_sentence])
    print("ENBATCH DECODED : ", [tokenizer_en.id_to_piece(id) for id in en_sentence])

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length)
val_dataset = val_dataset.padded_batch(BATCH_SIZE, padded_shapes=([MAX_LENGTH], [MAX_LENGTH]))