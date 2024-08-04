import argparse
import json

import pandas as pd
import numpy as np

from app.recommendations import RecommenderSystem


def precision_at_k(recommended_items, relevant_items, k):
    recommended_at_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    return len(recommended_at_k & relevant_set) / k


def evaluate_recsys(
    metrics_savepath,
    val_ratings_path, 
    faiss_index_path, 
    db_path,
    n_recommend_items,
):
    recsys = RecommenderSystem(
        faiss_index_path=faiss_index_path, 
        db_path=db_path)

    val_ratings = pd.read_csv(val_ratings_path)
    grouped_items = val_ratings.groupby("user_id")["item_id"].apply(list).reset_index()
    grouped_items = grouped_items["item_id"].tolist()


    metric_arrays = {
        "precision@1": [],
        "precision@3": [],
        "precision@10": []
    }

    for item_group in grouped_items:
        if len(item_group) == 1:
            continue

        ### Precision@k is computed for each edge.
        ### We will first aggregate it over all edges for user
        ### And after that - aggregate over all users
        user_metric_arrays = dict()
        for metric in metric_arrays.keys():
            user_metric_arrays[metric] = []

        for item in item_group:
            recommend_items = list(recsys.recommend_items(item, n_recommend_items))
            relevant_items = set(item_group) - {item}

            user_metric_arrays["precision@1"].append(
                precision_at_k(recommend_items, relevant_items, k=1))
            user_metric_arrays["precision@3"].append(
                precision_at_k(recommend_items, relevant_items, k=3))
            user_metric_arrays["precision@10"].append(
                precision_at_k(recommend_items, relevant_items, k=10))

        for metric in metric_arrays.keys():
            user_metric = np.mean(user_metric_arrays[metric])
            metric_arrays[metric].append(user_metric)

    metrics = dict()
    for metric, array in metric_arrays.items():
        metrics[metric] = np.mean(array)

    with open(metrics_savepath, "w") as f:
        json.dump(metrics, f)
    print(f"Saved metrics to {metrics_savepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a recommendation system.")
    parser.add_argument("--metrics_savepath", required=True, type=str, help="Path to save the evaluation metrics.")
    parser.add_argument("--val_ratings_path", required=True, type=str, help="Path to the csv file with validation ratings.")
    parser.add_argument("--faiss_index_path", required=True, type=str, help="Path to the FAISS index.")
    parser.add_argument("--db_path", required=True, type=str, help="Path to the database file.")
    parser.add_argument("--n_recommend_items", type=int, default=10, help="Number of items to recommend.")
    args = parser.parse_args()
    evaluate_recsys(**vars(args))