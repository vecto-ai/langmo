from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("/work/data/NLP/corpora/raw_texts/Eng/BNC").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=5, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save(".", "bnc")
