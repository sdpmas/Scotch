---
title: How does funcdef works? 
permalink: workings
key: docs-workings
---

## Introduction:

funcdef is a semantic code search tool. Most of the general-purpose search engines utilize keyword or textual matching on the query and index to get search results. This includes code search engines like [searchcode](https://searchcode.com/). However, such matching ignores all the semantic structure in code, which gives code a meaning instead of certain keywords present in the code or comments around the code. funcdef uses neural net to capture semantic relationships between and within query and code by representing both modalities in the same vector space.

The workings of funcdef can be described in following three steps:

### Data Collection:

Currently, funcdef only supports search over GitHub. Firstly, function definitions are extracted from the GitHub repositories of respective languages (Python, Javascript, Java, and Go as of now). After filtration and deduplication, the remaining set of function definitions forms a codebase to search over. The dataset of such functions is open-sourced as [funcdef dataset](https://huggingface.co/datasets/Samip/funcdef) in Hugging Face Datasets.

### Model:

Following [CodeXGLUE](https://arxiv.org/pdf/2102.04664.pdf), we train a CodeBERT model to align (NL, Code) pairs from the funcdef dataset. We use the same CodeBERT model to encode both code and NL. The representation of CLS token is regarded as the high-dimensional representation of code and NL. Refer to [CodeXGLUE](https://arxiv.org/pdf/2102.04664.pdf) for more details. 

### Search:

We use the high-dimensional representations of functions in funcdef dataset (after some filtrations) as the index to search over. Given a query, it gets encoded into the vector representation using CodeBERT model, and [ScaNN algorithm](https://github.com/google-research/google-research/tree/master/scann) is used to calculate vector similarity between functions in the funcdef dataset and the query. Functions with high similarities are returned as the search results.





