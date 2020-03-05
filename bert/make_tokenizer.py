from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, BertWordPieceTokenizer

# path_corpus = "/home/blackbird/Cloud/remote/berlin/data/NLP/corpora/raw_texts/Eng/BNC"
# path_corpus = "/work/data/NLP/corpora/raw_texts/Eng/BNC"
# path_corpus = "/home/blackbird/Projects_heavy/NLP/langmo/data/sample"
path_corpus = "/home/blackbird/Projects_heavy/NLP/langmo/data/wikitext-2"
paths = [str(x) for x in Path(path_corpus).glob("**/*.txt")]

# Initialize a tokenizer
#tokenizer = ByteLevelBPETokenizer()
tokenizer = CharBPETokenizer()
#tokenizer = BertWordPieceTokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=30000, min_frequency=5, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
# tokenizer.train(files=paths, vocab_size=10000, min_frequency=1, special_tokens=[])

# Save files to disk
tokenizer.save(".", "bnc")
r = tokenizer.encode("Hi, how do you do NoNeXistingTkn?")
print(r.tokens)
