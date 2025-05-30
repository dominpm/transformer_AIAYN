import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='/home/asidpm/transformer_dpm/transformer_AIAYN/artifacts/corpus_and_tokenizers/corpus_aztr.txt', model_prefix='aztr', vocab_size=10000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>'
)

spm.SentencePieceTrainer.train(
    input='/home/asidpm/transformer_dpm/transformer_AIAYN/artifacts/corpus_and_tokenizers/corpus_en.txt', model_prefix='en', vocab_size=10000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>'
)

sp_aztr = spm.SentencePieceProcessor(model_file='aztr.model')
print(f"BOS ID: {sp_aztr.bos_id()} → {sp_aztr.id_to_piece(sp_aztr.bos_id())}")
print(f"EOS ID: {sp_aztr.eos_id()} → {sp_aztr.id_to_piece(sp_aztr.eos_id())}")
print(f"PAD ID: {sp_aztr.pad_id()} → {sp_aztr.id_to_piece(sp_aztr.pad_id())}")
print(f"UNK ID: {sp_aztr.unk_id()} → {sp_aztr.id_to_piece(sp_aztr.unk_id())}")

sp = spm.SentencePieceProcessor(model_file='en.model')
print(f"BOS ID: {sp.bos_id()} → {sp.id_to_piece(sp.bos_id())}")
print(f"EOS ID: {sp.eos_id()} → {sp.id_to_piece(sp.eos_id())}")
print(f"PAD ID: {sp.pad_id()} → {sp.id_to_piece(sp.pad_id())}")
print(f"UNK ID: {sp.unk_id()} → {sp.id_to_piece(sp.unk_id())}")


def analyze_tokenizer(model_path, corpus_path, num_samples=5):
    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=model_path)

    # Print special token IDs
    print("🔍 Special Tokens:")
    print(f"PAD ID: {sp.pad_id()} → {sp.id_to_piece(sp.pad_id())}")
    print(f"UNK ID: {sp.unk_id()} → {sp.id_to_piece(sp.unk_id())}")
    print(f"BOS ID: {sp.bos_id()} → {sp.id_to_piece(sp.bos_id())}")
    print(f"EOS ID: {sp.eos_id()} → {sp.id_to_piece(sp.eos_id())}")

    # Read corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_tokens = 0
    unk_tokens = 0
    total_lines = len(lines)

    for line in lines:
        pieces = sp.encode(line, out_type=int)
        total_tokens += len(pieces)
        unk_tokens += pieces.count(sp.unk_id())

    coverage = 100 * (1 - unk_tokens / total_tokens) if total_tokens else 0

    print("\n📊 Tokenizer Coverage Analysis:")
    print(f"- Total lines analyzed: {total_lines}")
    print(f"- Total tokens: {total_tokens}")
    print(f"- Unknown tokens: {unk_tokens}")
    print(f"- Coverage: {coverage:.2f}%")

    # Sample tokenizations
    print("\n🔠 Sample Tokenizations:")
    for line in lines[:num_samples]:
        print(f"Original: {line}")
        ids = sp.encode(line, out_type=int)
        tokens = [sp.id_to_piece(i) for i in ids]
        print(f"Tokens:   {tokens}")
        print("-" * 60)

# Example usage
analyze_tokenizer('aztr.model', '/home/asidpm/transformer_dpm/transformer_AIAYN/artifacts/corpus_and_tokenizers/corpus_aztr.txt')
analyze_tokenizer('en.model', '/home/asidpm/transformer_dpm/transformer_AIAYN/artifacts/corpus_and_tokenizers/corpus_en.txt')
