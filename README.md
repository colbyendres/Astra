![Astra](static/assets/astra.png)
# Astra: An arXiv-based Paper Recommendation System

## Overview
ASTRA (ArXiv Sourced Text Recommendation Agent), as the name suggests, is a paper recommendation service based on arXiv. We've collected thousands of papers across various scientific disciplines for 
recommendation. Simply provide a paper's title, arXiv ID, or a handful of keywords, and Astra will curate a list of relevant documents.

## How does Astra work?
Like many RAG-style applications, Astra relies on quality embeddings of documents and user queries. The core component of our recommendation engine is [Embed](https://cohere.com/embed), 
a large language model developed by Cohere. Embed was built to handle mixed-modality documents and has demonstrated results in retrieval, making it a perfect choice for Astra. With this approach, 
recommendation is simply a k-nearest neighbors problem, made expedient with the help of [FAISS](https://github.com/facebookresearch/faiss/wiki).

## How to Use
You can find our deployment of Astra [here](https://astra-recommender-ba35341bcf67.herokuapp.com). If you intend on building Astra locally:
```
git clone https://github.com/colbyendres/Astra/
cd Astra
pip install -r requirements.txt
```
which pulls in all the necessary dependencies. For generating the embedding datasets, take a look at the `notebooks` directory. With the database seeded, simply run with `flask run`.
