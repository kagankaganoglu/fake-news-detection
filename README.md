# Fake News Detection on LIAR (CS445 · Group 32)

# #A 6-class truthfulness classifier on the LIAR dataset using two approaches:
(1) a BiLSTM with pre-trained GloVe embeddings, and (2) a hybrid CNN that ingests BERT text embeddings plus speaker/metadata with an attention-based fusion. 

## Dataset

We use the LIAR dataset (PolitiFact statements) with six labels: pants-fire, false, barely-true, half-true, mostly-true, true. Splits follow the original train/valid/test partitions. 

## Methods

BiLSTM (GloVe-100d): classic sequence model for text.

Hybrid CNN (BERT + metadata): multi-kernel CNN over text embeddings; dense stack for metadata; attention to fuse modalities; regularized MLP head. 

## Results (summary)

BiLSTM outperformed LR/SVM/NB baselines on accuracy and macro-F1. 

Hybrid CNN peaked at ~26.85% validation accuracy (epoch 33) and struggled across similar labels, reflecting class imbalance. 
