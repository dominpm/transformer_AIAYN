import tensorflow as tf
from dataset import MAX_LENGTH, tokenizer_aztr, tokenizer_en
# from main import automated_translation_transformer

from transformer import Transformer
from main import params_config

def translate(sentence: str,
              max_length: int = MAX_LENGTH,
              src_tokenizer=tokenizer_aztr,
              tgt_tokenizer=tokenizer_en,
              model=None):

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
    for _ in range(max_length - 1):
        predictions = model(encoder_input, decoder_input, training=False)
        next_token_logits = predictions[:, -1, :]
        next_token_id = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
        next_token_id = tf.expand_dims(next_token_id, -1)

        # Imprimir el token generado
        token_int = int(next_token_id.numpy()[0][0])
        token_str = tgt_tokenizer.decode([token_int])
        print(token_str, end=' ', flush=True)

        decoder_input = tf.concat([decoder_input, next_token_id], axis=-1)

        if token_int == EOS_ID:
            break


    result_ids = decoder_input.numpy()[0][1:].tolist()
    if result_ids and result_ids[-1] == EOS_ID:
        result_ids = result_ids[:-1]

    return tgt_tokenizer.decode(result_ids)

model_path = './artifacts/transformer/train_checkpoints/weights/weights_model_epoch01_valacc0.2477.weights.h5'
automated_translation_transformer = Transformer(**params_config)
dummy_encoder_input = tf.zeros((1, 10), dtype=tf.int32)
dummy_decoder_input = tf.zeros((1, 9), dtype=tf.int32)
_ = automated_translation_transformer(dummy_encoder_input, dummy_decoder_input, training=False)
automated_translation_transformer.load_weights(model_path)


def main():
    print("Traductor automático AZ-TR -> EN")
    while True:
        sentence = input("Introduce una frase en Azerí/Turco (o 'salir' para terminar): ")
        if sentence.lower() == 'salir':
            break

        translated_sentence = translate(
            sentence=sentence,
            max_length=MAX_LENGTH,
            src_tokenizer=tokenizer_aztr,
            tgt_tokenizer=tokenizer_en,
            model=automated_translation_transformer
        )

        print(f"Traducción: {translated_sentence}\n")

if __name__ == "__main__":
    main()