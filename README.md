# Overview

The entity matching process breaks down into two steps: blocking and matching.

## Blocking
After cleaning and standardizing the data in both the FERC and EIA datasets, we perform a process called blocking, in which we remove record pairs that are unlikely to be matched from the candidate set of record pairs. This reduces computational complexity and boosts model performance, as we no longer need to evaluate n<sup>2</sup> candidate match pairs and instead only evaluate a set of record pairs that are more likely to be matched. The goal of blocking is to create a set of candidate record pairs that is as small as possible while still containing all correctly matched pairs.

### Rule Based Blocking
The simplest way we tried to create blocks of candidate pairs is with rule based blocking. This involves creating a set of heuristics that, when applied disjunctively, create "blocks" of record pairs that form a complete candidate set of pairs. This approach was too simple for our problem, and it was difficult to capture the training data matches without creating a very large candidate set.

It's worth noting that the output of rule based blocking can be combined with the output of an embedding vector approach described below to increase recall, while increasing the blocking output size only modestly (Thirumuruganathan, Li).

### Distributed Representations for Blocking
Instead of creating heuristics for blocking, we can create embedding vectors that represent the tuples in the FERC and EIA datasets and find the most similar pairs of embedding vectors to create a candidate set. This process involves three main steps.  
1. Attribute Embedding: For each tuple `t` in the FERC and EIA datasets, compute an embedding vector for each attribute (column) in `t`. This can sometimes involve vectorizing each individual word or n-gram within an attribute, and then combining them into one attribute embedding.
2. Tuple Embedding: Combine each attribute embedding vector into one embedding vector for the tuple `t`.
3. Vector Pairing: Find similar vector pairs from the FERC and EIA datasets using a similarity metric and add the tuple pairs represented by these embedding vectors to the candidate set.

**Word and Attribute Embeddings**

There are multiple methods for embedding the string value attributes of the tuples. 

- TF-IDF: 
  - Vectorizes based on the occurrence of words in the domain
  - Can't control the output dimension of the embeddings

Vectorizing a single word or n-gram creates a "word embedding" and then these word embeddings can be combined together (see below section) into "attribute embeddings" for each column.

- Word Embeddings (word2vec, GloVe):
  - Can be trained on the domain instead of pre-trained
  - Better represents the relationships between words and groups similar words close in vector space. i.e. the relationship between "king" and "queen"

- Character Level (fastText) or Sub-Word Embedding (bi-grams):
  - Can handle similarities in words like "generator", "generation"
  - Better handles typos

The numeric attributes can be normalized within each column. (or should they go through the same embedding process as the string columns? in the case of TF-IDF does it matter if the numeric columns aren't on the same scale as the string columns?)

**Tuple Embedding**

- Equal Weight Aggregation: In the case of word embeddings like `word2vec`, all the word embeddings within an attribute are averaged together to create an attribute embedding and then all attribute embeddings are concatenated together into one tuple embedding. So if there are `m` attributes and the dimension of each attribute embedding is `d`, then the tuple embedding would be lenght `m x d`. Used in DeepER (Ebraheem, Thirumuruganathan)

- Weighted Aggregation: Same as equal weight aggregation, but a weighted average is used to combine the word embeddings together into an attribute embedding, and then concatenated into one tuple embedding. The weights of the attribute embeddings can optionally be learned.

Note: With aggregation methods, order is not considered: "Generator 10" has the same embedding as "10 Generator" (could be good or bad)

- LSTM or RNN: Used in DeepER (Ebraheem, Thirumuruganathan). Considers the order of words as well as the order and relationship of attributes in a tuple.

- Self-Reproduction: Autoencoder or seq2seq. From "Deep Learning for Blocking in Entity Matching: A Design Space Exploration" (Thirumuruganathan, Li)

_Roughly speaking, they take a tuple t, feed it into a neural network (NN) to output a compact embedding vector u<sub>t</sub>, such that if we feed u<sub>t</sub> into a second NN, we can recover the original tuple t (or a good approximation of t). If this happens, u<sub>t</sub> can be viewed as a good compact summary of tuple t, and can be used as the tuple
embedding of t._

<img width="607" alt="Screen Shot 2023-03-23 at 12 31 44 PM" src="https://user-images.githubusercontent.com/95320980/227329550-ae77a54a-8526-44b6-8a71-2d8137dda70a.png">


**Vector Pairing**

- Similarity-based: cosine similarity and Euclidean distance
  - either only keep tuple pairs where the similarity is above a threshold (e.g. .7)
  - or keep kNN for each tuple
- Hash-based: Locality Sensitive Hashing
  - more efficient than similarity-based
  - hash tuple embeddings and only keep tuples that have the same hash value
  - LSH hashes similar vectors into the same bucket with high probability (used in DeepER, AutoBlock)

**Evaluation Metric**
- Reduction Ratio (percentage that the candidate set has been reduced from n x n comparisons)
- Pairs Completeness (percentage of matching record pairs contained within the reduced comparison space after blocking)
- Harmonic Mean of Reduction Ratio and Pairs Completeness: 2 * RR * PC / (RR + PC)

These metrics work best for a rules based blocking method, where you can't adjust the size of the candidate set. Include metrics for blocking when Vector Pairing step is done at end to retain k most similar vector pairs.

**Experiment Matrix**
(Note: There could probably be more experimentation added with the way that numeric attributes are embedded and concatenated onto the record tuple embedding)

|Attribute Embedding Method|Tuple Embedding Method|% of Training Matches Retained|
|-----------------------------|-------------------------|--------------------------------|
|Rule Based Blocking|||
|TF-IDF|Equal Weight Aggregation||
||Weighted Aggregation||
||autoencoder||
||seq2seq||
|word2vec|Equal Weight Aggregation||
||Weighted Aggregation||
||autoencoder||
||seq2seq||
|fastText|Equal Weight Aggregation||
||Weighted Aggregation||
||autoencoder||
||seq2seq||


# Repo Overview

The `labeling_function_notebooks` directory contains downloaded labeling function notebooks and logs from Panda. The notebooks are labeled with numbers that correspond to the CSV of exported matched records generated from this notebook. The CSV's of matched records generated from these notebooks are contained in the `panda_matches` directory.

I've added the 2020 input zip file I've been using for Panda in the `panda_inputs` directory of this repo. This zip file contains a left and right CSV where `left.csv` is the 2020 FERC 1 records and `right.csv` is the distinct (only true granularities) 2020 EIA plant parts records. These inputs maybe shouldn't be checked-in in the future, but are nice for now because it takes a long time to generate the plant parts list and then run the preprocessing function.

`train_ferc1_eia.csv` is the training data labels of matched EIA and FERC records, taken from the RMI repo. The 2020 records are separated out from this in `train_ferc1_eia_2020.csv`. I need to check with people working on RMI stuff to make sure this is the most up to date training data.

The notebook `training_data_metrics` contains code to compare the Panda found matches with the hand labeled training data matches.
The notebook `preprocessing` contains code to do various preprocessing things like generate the updated plant parts list, make it distinct, add on utility name, etc. After this pre-processing step is done, the data is ready for the blocking step.

Panda only found 24 of the 115 hand-labeled matches for 2020. Compared to the baseline model, Panda finds 203 of the 1151 matches found by the baseline model for 2020. Panda finds around 2800 matches total for 2020. I take a deeper look at this in the `training_data_metrics` notebook. I think this low recall is most likely because of NaN values.
