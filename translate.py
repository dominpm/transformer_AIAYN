import tensorflow as tf
from dataset import MAX_LENGTH, tokenizer_aztr, tokenizer_en
from main import automated_translation_transformer

def translate(sentence: str,
              max_length: int = MAX_LENGTH,
              src_tokenizer=tokenizer_aztr,
              tgt_tokenizer=tokenizer_en,
              model=automated_translation_transformer):

    BOS_ID = tgt_tokenizer.bos_id()
    EOS_ID = tgt_tokenizer.eos_id()

    if BOS_ID == -1:
        BOS_ID = tgt_tokenizer.piece_to_id('<s>')
    if EOS_ID == -1:
        EOS_ID = tgt_tokenizer.piece_to_id('</s>')

    encoder_input = src_tokenizer.encode(sentence)[:max_length]
    encoder_input = tf.expand_dims(encoder_input, 0)          # (1, seq_len)

    decoder_input = tf.expand_dims([BOS_ID], 0)               # (1, 1)

    # Bucle Greedy (AUTOREGRESIVO)
    for _ in range(max_length - 1):                           # ya tenemos 1 token
        # forward
        predictions, _ = model(encoder_input, decoder_input, training=False)

        next_token_logits = predictions[:, -1, :]
        next_token_id = tf.argmax(next_token_logits,
                                  axis=-1,
                                  output_type=tf.int32)
        next_token_id = tf.expand_dims(next_token_id, -1)

        decoder_input = tf.concat([decoder_input, next_token_id], axis=-1)

        if int(next_token_id.numpy()[0][0]) == EOS_ID:
            break

    result_ids = decoder_input.numpy()[0][1:]
    if result_ids and result_ids[-1] == EOS_ID:
        result_ids = result_ids[:-1]

    return tgt_tokenizer.decode(result_ids)