#!/bin/bash
set -e

input_directory="$1"
save_directory="$2"
device="${3:-cpu}"

echo Running on "$device".

PYTHONPATH=. python exp/process_raw_data.py \
    --input_directory "$input_directory" \
    --save_directory "$save_directory" \
    --create_train_val_split

PYTHONPATH=. python exp/deepwalk.py \
    --items_path "$save_directory/items.csv" \
    --ratings_path "$save_directory/train_ratings.csv" \
    --embeddings_savepath "$save_directory/deepwalk_embeddings.npy" \
    --device $device \
    --no_wandb

PYTHONPATH=. python exp/sbert.py \
    --items_path "$save_directory/items.csv" \
    --embeddings_savepath "$save_directory/text_embeddings.npy" \
    --device $device

PYTHONPATH=. python exp/gnn.py \
    --items_path "$save_directory/items.csv" \
    --ratings_path "$save_directory/train_ratings.csv" \
    --text_embeddings_path "$save_directory/text_embeddings.npy" \
    --deepwalk_embeddings_path "$save_directory/deepwalk_embeddings.npy" \
    --embeddings_savepath "$save_directory/embeddings.npy"\
    --device $device \
    --no_wandb 

PYTHONPATH=. python exp/prepare_index.py \
    --embeddings_path "$save_directory/embeddings.npy" \
    --save_path "$save_directory/index.faiss"

PYTHONPATH=. python exp/prepare_db.py \
    --items_path "$save_directory/items.csv" \
    --embeddings_path "$save_directory/embeddings.npy" \
    --db_path "$save_directory/items.db"

PYTHONPATH=. python exp/evaluate.py \
    --metrics_savepath "$save_directory/metrics.json" \
    --val_ratings_path "$save_directory/val_ratings.csv" \
    --faiss_index_path "$save_directory/index.faiss" \
    --db_path "$save_directory/items.db"

echo "Evaluation metrics:"
cat "$save_directory/metrics.json"