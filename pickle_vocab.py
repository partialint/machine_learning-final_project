#!/usr/bin/env python3
import pickle
from collections import Counter

def build_vocab(pos_file, neg_file, output_file):
    words = []
    
    for file_path in [pos_file, neg_file]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words.extend(line.strip().split())
    
    word_counts = Counter(word for word in words if word.strip())
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in sorted(word_counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{count}\t{word}\n")

def process_vocab(input_file, output_file):
    words = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lstrip()
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            
            count, word = parts
            count = int(count)
            
            if count > 4:
                words.append((count, word.strip()))
    
    words.sort(reverse=True, key=lambda x: x[0])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for count, word in words:
            f.write(f"{word}\n")






def main():
    build_vocab(
    pos_file=r"public_kaggle_files/twitter-datasets/train_pos_full.txt",
    neg_file=r"public_kaggle_files/twitter-datasets/train_neg_full.txt",
    output_file="vocab_full.txt"
)
    
    process_vocab(
    input_file="vocab_full.txt",
    output_file="vocab_cut.txt"
)
    
    vocab = dict()
    with open("vocab_cut.txt") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
