#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_tweets(filename, vocab=None):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]
    
def text_to_vector(text, embeddings, vocab):
    words = text.strip().split()
    vectors = []
    for word in words:
        if word in vocab:
            idx = vocab[word]
            vectors.append(embeddings[idx])
    return np.mean(vectors, axis=0) if vectors else np.zeros(embeddings.shape[1])

def train_classifier(embeddings, vocab):
    """训练并保存分类器"""
    pos_texts = load_tweets("public_kaggle_files/twitter-datasets/train_pos.txt")
    neg_texts = load_tweets("public_kaggle_files/twitter-datasets/train_neg.txt")

    X = []
    y = []

    print("Converting texts to vectors...")
    for text in pos_texts:
        X.append(text_to_vector(text, embeddings, vocab))
        y.append(1)

    for text in neg_texts:
        X.append(text_to_vector(text, embeddings, vocab))
        y.append(-1)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print("Training classifier...")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    print(f"Train accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}")

    with open("classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    
    return clf

def main():
    print("loading cooccurrence matrix")
    with open("cooc.pkl", "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))
    

    nmax = 100
    embedding_dim = 100
    eta=0.001
    initial_eta = 0.01
    base_eta = 0.012
    alpha = 3 / 4
    xmax = 100
    epochs = 50000
    batch_size = 248576
    warmup_epochs=20000

    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    xs = torch.randn(cooc.shape[0], embedding_dim, device=device)
    ys = torch.randn(cooc.shape[1], embedding_dim, device=device)
    bx = torch.zeros(cooc.shape[0], device=device)
    by = torch.zeros(cooc.shape[1], device=device)
    
    
    

    
    cooc = cooc.tocoo()
    rows = torch.LongTensor(cooc.row).to(device)
    cols = torch.LongTensor(cooc.col).to(device)
    data = torch.FloatTensor(cooc.data).to(device)

    weights = torch.where(data < xmax, 
                         (data / xmax) ** alpha, 
                         torch.ones_like(data))
    
    num_samples = len(data)
    indices = torch.randperm(num_samples, device=device)


    for epoch in range(epochs):

        
        if epoch < warmup_epochs:
            eta = initial_eta + (base_eta - initial_eta) * (epoch / warmup_epochs)
        else:
            eta = base_eta * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

        # print("epoch {}".format(epoch))
        cost = 0
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_rows = rows[batch_indices]
            batch_cols = cols[batch_indices]
            batch_data = data[batch_indices]
            batch_weights = weights[batch_indices]
            
            # Vectorized computations
            x_emb = xs[batch_rows]
            y_emb = ys[batch_cols]
            x_bias = bx[batch_rows]
            y_bias = by[batch_cols]
            
            predictions = (x_emb * y_emb).sum(dim=1) + x_bias + y_bias
            errors = torch.log(batch_data) - predictions
            
            # Compute gradients
            grad_scaling = batch_weights * errors
            grad_x = grad_scaling.unsqueeze(1) * y_emb
            grad_y = grad_scaling.unsqueeze(1) * x_emb
            grad_bx = grad_scaling
            grad_by = grad_scaling
            
            # Update parameters
            xs.index_add_(0, batch_rows, grad_x * eta)
            ys.index_add_(0, batch_cols, grad_y * eta)
            bx.index_add_(0, batch_rows, grad_bx * eta)
            by.index_add_(0, batch_cols, grad_by * eta)
            
            cost += (batch_weights * errors ** 2).sum().item()

        embeddings = ((xs + ys) / 2).cpu().numpy()

        print("Epoch {}, cost: {}".format(epoch, cost / cooc.nnz))

    # fill in your SGD code here,
    # for the update resulting from co-occurence (i,j)

        np.save("embeddings.npy", embeddings)
    
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    train_classifier(embeddings, vocab)

        
        

    

if __name__ == "__main__":
    main()
