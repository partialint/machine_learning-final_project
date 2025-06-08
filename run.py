import os
import subprocess
import numpy as np
import pickle
import csv

def predict_sentiment(text, embeddings, vocab, clf):
    words = text.strip().split()
    vectors = []
    for word in words:
        if word in vocab:
            idx = vocab[word]
            vectors.append(embeddings[idx])
    vec = np.mean(vectors, axis=0) if vectors else np.zeros(embeddings.shape[1])
    return clf.predict([vec])[0]


def predict_test_file(input_file, output_file, embeddings, vocab, clf):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        
        writer = csv.writer(f_out)
        writer.writerow(['Id', 'Prediction'])
        
        for line in f_in:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
                
            tweet_id, text = parts
            text = text.lstrip()
            sentiment = predict_sentiment(text, embeddings, vocab, clf)
            writer.writerow([tweet_id, sentiment])


def main():
    subprocess.run(["python3", "pickle_vocab.py"])
    
    subprocess.run(["python3", "cooc.py"])
    
    subprocess.run(["python3", "glove_template.py"])
    
    embeddings = np.load("embeddings.npy")
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    test_texts = [
        "It's terrible",
        "I'm okay",
        "This is good"
        
    ]

    for text in test_texts:
        sentiment = predict_sentiment(text, embeddings, vocab, clf)
        print(f"Text: '{text}' -> Sentiment: {sentiment} (1=positive, -1=negative)")

    predict_test_file(
        input_file="public_kaggle_files/twitter-datasets/test_data.txt",
        output_file="my_submission.csv",
        embeddings=embeddings,
        vocab=vocab,
        clf=clf
    )
    print("Predictions saved to my_submission.csv")

if __name__ == "__main__":
    main()