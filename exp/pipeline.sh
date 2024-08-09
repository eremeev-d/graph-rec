#!/bin/bash
set -e

input_directory="$1"
save_directory="$2"
device="${3:-cpu}"

echo Running on "$device".

PYTHONPATH=. python exp/process_raw_data.py \
    --input_directory "$input_directory" \
    --save_directory "$save_directory" \
    --create_train_val_test_split

PYTHONPATH=. python exp/sbert.py \
    --items_path "$save_directory/items.csv" \
    --embeddings_savepath "$save_directory/text_embeddings.npy" \
    --device $device

PYTHONPATH=. python exp/gnn/train.py \
    --items_path "$save_directory/items.csv" \
    --train_ratings_path "$save_directory/train_ratings.csv" \
    --val_ratings_path "$save_directory/val_ratings.csv" \
    --text_embeddings_path "$save_directory/text_embeddings.npy" \
    --embeddings_savepath "$save_directory/embeddings.npy"\
    --model_savepath "$save_directory/model.pt" \
    --device $device \
    --no_wandb

PYTHONPATH=. python exp/prepare_recsys.py \
    --items_path "$save_directory/items.csv" \
    --embeddings_path "$save_directory/embeddings.npy" \
    --save_directory "$save_directory"

PYTHONPATH=. python exp/evaluate.py \
    --metrics_savepath "$save_directory/metrics.json" \
    --val_ratings_path "$save_directory/test_ratings.csv" \
    --faiss_index_path "$save_directory/index.faiss" \
    --db_path "$save_directory/items.db"

echo "Evaluation metrics:"
cat "$save_directory/metrics.json"