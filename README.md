---
title: A simple Graph-based Recommender System
emoji: 📚
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 8501
---
# A simple Graph-based Recommender System

### What is it?

This app is a simple graph-based recommender system that searches for items and recommends similar ones. It can be applied to any dataset. For demonstration purposes, we use the (filtered) [Goodreads](https://mengtingwan.github.io/data/goodreads#datasets) dataset.

### Where can I try this app?

The app is currently deployed at HuggingFace Spaces ([link](https://huggingface.co/spaces/eremeev-d/graph-rec)). You will probably need to wait a minute or two for app to start running. 

### How to use it?

Simply enter a keyword (e.g., "Brave") into the search bar and press the "Search" button. The app will display relevant books along with their short descriptions.

For each book, you can click "Recommend Similar Items" to see other books you might enjoy if you liked the selected one.

### How to reproduce embeddings computation?

First, install needed requirements from `exp/requirements.txt` (or `exp/requirements_gpu.txt` for gpu) file. 

Then, download needed raw data from [Goodreads website](https://mengtingwan.github.io/data/goodreads#datasets). We will need the following files: `book_id_map.csv`, `goodreads_books.json`, `goodreads_interactions.csv` and `user_id_map.csv`. You can download this files manually or use this [Kaggle dataset](https://www.kaggle.com/datasets/eremeevd/graph-rec-goodreads). 

Finally, simply run the following command at the root of the repo:
```
sh exp/prepare_embeddings.sh INPUT_DIRECTORY SAVE_DIRECTORY
```
where `INPUT_DIRECTORY` is path to the directory with raw data (e.g. `/kaggle/input/graph-rec-goodreads/goodreads-books`). And `SAVE_DIRECTORY` is path to the directory, where results will be saved (e.g. `/kaggle/working/embeddings`). To use obtained embeddings, copy the following files to the `app/data`: `index.faiss` and `items.db`.

To run on GPU, run the following command:
```
sh exp/prepare_embeddings.sh INPUT_DIRECTORY SAVE_DIRECTORY cuda
```

For further information, refer to the `exp` directory in this repo.