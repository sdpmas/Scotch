---
title: How does Scotch work? 
permalink: workings
key: docs-workings
---

## Introduction:

Most general-purpose code search engines utilize keyword or textual matching between the query (natural language intent) and index to get search results. However, such matching ignores the semantic structure in code that represents the intent of the snippet and only utilizes keywords in the code or surrounding comments. Scotch is a semantic code search tool that uses neural networks to capture semantic relationships between the natural language query and code by representing both modalities in the same vector space.
The workings of Scotch can be described in the following three steps:


### Data Collection:

Firstly, function definitions for various languages (Python, Javascript, Java, and Go) are extracted from GitHub repositories. After filtration and deduplication, the remaining set of function definitions forms a codebase to search over. This dataset, which currently contains about 13M functions from four languages, will be open-sourced as the Scotch dataset. Out of the 13M functions, about 4M functions have corresponding docstrings. The dataset is split into train, test and valid sets containing 80, 10, and 10 percent of the dataset respectively. We use only the functions with docstrings to train the code search model. The search is conducted over all functions in the Scotch dataset after some heuristic-based filtering. Currently, Scotch supports search over GitHub only.

### Model:

Following [CodeXGLUE](https://arxiv.org/pdf/2102.04664.pdf), we train a [CodeBERT](https://arxiv.org/abs/2002.08155) model to align pairs of natural language (NL) intent and code. For this purpose, we assume that the docstring represents NL intent and use the (docstring, function) pairs from the Scotch dataset. We use the same CodeBERT model to encode both code and NL intent. The representation of CLS token is treated as the high-dimensional vector representation of code and NL for indexing. See [CodeXGLUE](https://arxiv.org/pdf/2102.04664.pdf) for more details.


### Search:

We use the high-dimensional representations of functions in Scotch dataset (after some filtrations) as the index to search over. Given a query, it gets encoded into the vector representation using CodeBERT model, and [ScaNN algorithm](https://github.com/google-research/google-research/tree/master/scann) is used to calculate vector similarity between functions in the Scotch dataset and the query. Functions with high similarities are returned as the search results.






