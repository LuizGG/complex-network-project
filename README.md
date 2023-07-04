# complex-network-project
Final project for SME5924 part of the masters program at USP (Universidade de São Paulo)

## Problem description

Extract network attributes to serve as features on a machine learning task. To do so, we explore the bank customer churn dataset available on [kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn).

It is a tabular dataset with customer attributes and a binary flag indicating if the customer churned or not. It has 15 features, 10k rows and about 20% incidence of churn

## Creating a network

A network is created by computing the pair-wise correlation between samples. One thing to note is that the original correlation matrix between the samples is very skewed. To circumvent this, PCA pre-processing is applied which centeres the correlation on 0 and will, eventually, produce smaller graphs to which we can compute the atttributes in quicker fashion.

In addition, to compute the correlation, we use `spearman` method.

![img](/imgs/01_correlation.png)

Finally, we split the dataset into training and validation, to which we will create two networks as to avoid any potential leak of information from the trian to the validation set. Therefore, the network created on the train set will produce features for the training dataset, while another network, created on the full dataset, will provide the features for the validation set.

With that, we then decided on computing three networks considering a connection whenever the absolute correlation between two samples is higher than or equal to 0.7, 0.8 and 0.9. The attributes are extracted at node level and are:
* Node degree
* Average neighbours degree
* Betweeness, closeness and eigenvector centrality
* Clustering coefficient

Yielding 6 new features for each threshold, producing a total of 18 new features.

### Node2Vec

We also create node level embeddings applying Node2Vec with the default parameters and an embedding size of 32. The technique is applied using the library [node2vec](https://github.com/eliorc/node2vec)

## Results

Different features sets are tested:
1. Considering the original dataset features
2. Attributes + Node2Vec for 0.7 threshold
3. Attributes + Node2Vec for 0.8 threshold
4. Attributes + Node2Vec for 0.9 threshold
5. Attributes + Node2Vec for All threshold
6. All features

Models are trained using a Random Forest with 100 trees and evaluated using ROC-AUC. The best model clearly is the one considering the original set, while, the model considering only the features from 0.7 threshold are the best from the set that looks only at network features.

![img](/imgs/02_results.png)

On the model fitted using all features, feature importance  shows that after the two best features from the original set, the next best are the centrality metrics from the 0.7 network. So on training time the model deems these information important but it fails to extrapolate to the validation set.

![img](/imgs/03_feature_importance.png)
