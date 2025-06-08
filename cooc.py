#!/usr/bin/env python3
from scipy.sparse import coo_matrix
from collections import defaultdict
import numpy as np
import pickle

def main():
    cooc_counts = defaultdict(int)
    
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print("Building co-occurrence matrix...")
    
    counter = 1
    for fn in ["twitter-datasets/train_pos_full.txt", 
               "twitter-datasets/train_neg_full.txt"]:
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                for line in f:
                    tokens = [vocab.get(t, -1) for t in line.strip().split()]
                    tokens = [t for t in tokens if t >= 0]
                    
                    for i in range(len(tokens)):
                        for j in range(len(tokens)):
                           if tokens[i] <= tokens[j]:
                                cooc_counts[(tokens[i], tokens[j])] += 1
                    
                    if counter % 10000 == 0:
                        print(f"Processed {counter} lines")
                    counter += 1
        except FileNotFoundError:
            print(f"Error: File {fn} not found!")
            return

    print("Creating sparse matrix...")
    
    data, row, col = [], [], []
    for (i, j), count in cooc_counts.items():
        row.append(i)
        col.append(j)
        data.append(count)
        if i != j:
            row.append(j)
            col.append(i)
            data.append(count)
    
    vocab_size = len(vocab)
    cooc = coo_matrix((data, (row, col)), 
                      shape=(vocab_size, vocab_size),
                      dtype=np.float32)
    
    print("Summing duplicates...")
    cooc.sum_duplicates()
    
    with open("cooc.pkl", "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
    print("Co-occurrence matrix saved to cooc.pkl")

if __name__ == "__main__":
    main()
