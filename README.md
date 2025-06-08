# Sentiment Analysis of Twitter Data using GloVe and Logistic Regression

**Author**: Xujie Wang (wangxujie@westlake.edu.cn)
**Collaborator**: Shuhan Zhan (zhaoshuhan@westlake.edu.cn)

## 1. Project Overview

This project is for the Westlake University Machine Learning Course, Spring 2025. The primary goal is to address a text sentiment classification challenge: predicting if a tweet originally contained a positive `:)` or negative `:(` smiley, using only the text content.

This work involves the implementation of a full machine learning pipeline, including exploratory data analysis, feature processing, model implementation, and performance analysis to generate final predictions. The methodology is centered around generating word embeddings via the GloVe algorithm and subsequently training a linear classifier on these features.

## 2. Setup & Dependencies

To run this project, please ensure you have Python installed. The following steps will guide you through setting up the environment and data.

### 2.1. Dataset

Since the dataset is too large, we don't provide it. It can be downloaded from the course's official GitHub page:
`https://github.com/LINs-lab/course_machine_learning/tree/master/projects/project2/project_text_classification/public_kaggle_files`

After downloading `twitter-datasets.zip`, please unzip it. The resulting `twitter-datasets` folder should be placed in the root directory of this project. The expected file structure is as follows:
your_project_folder/

├── twitter-datasets/

│   ├── train_pos_full.txt

│   ├── train_neg_full.txt

│   └── test_data.txt

├── run.py

├── pickle_vocab.py

├── cooc.py

├── glove_template.py

└── README.md

### 2.2. Dependencies

All required Python libraries for this project are listed in the `requirements.txt` file. They can be installed using the following command:

```bash
pip install -r requirements.txt
```
External libraries are permitted for this project, provided they are properly cited in the report and documented in the code.

### 2.3. Run

We integrated all the contents of this project into run.py. You can use the command below and wait for the result.

```bash
python run.py
```

And `my_submission.csv` is generated, which is the final result of this project.
