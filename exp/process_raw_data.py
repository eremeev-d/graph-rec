import argparse
import os
import json

import pandas as pd
import numpy as np


def book_filter(book, ratings_count_threshold=10_000):
    try:
        if book["ratings_count"] == "":
            return False
        if int(book["ratings_count"]) < ratings_count_threshold:
            return False

        if book["description"] == "":
            return False
        
        if book["title"] == "":
            return False
        
        if book["title_without_series"] == "":
            return False

        possible_lang_codes = {"eng", "en-GB", "en-US"}
        if not book["language_code"] in possible_lang_codes:
            return False

        return True
    except Exception:
        return False


def process_raw_data_goodreads(input_directory, save_directory, positive_rating_threshold = 4.0):
    os.makedirs(save_directory, exist_ok=True)

    ### Process items
    columns = [
        "book_id",
        "description",
        "title_without_series",
    ]
    numeric_columns = [
        "book_id",
    ]
    
    items = []
    with open(os.path.join(input_directory, "goodreads_books.json"), "r") as f:
        for line in f:
            item = json.loads(line)
            if book_filter(item):
                items.append([item[col] for col in columns])
    items = pd.DataFrame(items, columns=columns)
    for col in numeric_columns:
        items[col] = pd.to_numeric(items[col])
    items["item_id"] = items.index
    items["title"] = items["title_without_series"]
    items.drop("title_without_series", axis=1, inplace=True)
    items.to_csv(os.path.join(save_directory, "items.csv"), index=False)

    ### Process ratings
    ratings = pd.read_csv(os.path.join(input_directory, "goodreads_interactions.csv"))

    book_id_map = pd.read_csv(os.path.join(input_directory, "book_id_map.csv"))
    csv_to_usual_map = dict(zip(book_id_map["book_id_csv"], book_id_map["book_id"]))
    usual_to_csv_map = dict(zip(book_id_map["book_id"], book_id_map["book_id_csv"]))

    book_ids = items["book_id"].unique()
    book_ids_csv = set([usual_to_csv_map[book_id] for book_id in book_ids])
    ratings = ratings[ratings["rating"] >= positive_rating_threshold]
    ratings = ratings[ratings["book_id"].isin(book_ids_csv)]

    book_to_item_id_map = dict(zip(items["book_id"], items["item_id"]))
    ratings["item_id"] = ratings["book_id"].map(csv_to_usual_map).map(book_to_item_id_map)
    
    user_ids = list(ratings["user_id"].unique())
    user_ids_map = dict(zip(user_ids, range(len(user_ids))))
    ratings["user_id"] = ratings["user_id"].map(user_ids_map)

    ratings.to_csv(os.path.join(save_directory, "ratings.csv"), index=False)


def create_train_val_test_split(ratings_path, train_savepath, val_savepath, test_savepath, seed=42):
    ratings = pd.read_csv(ratings_path)
    user_ids = ratings["user_id"].unique()

    rng = np.random.default_rng(seed=seed)
    train_size = int(len(user_ids) * 0.7)
    val_size = int(len(user_ids) * 0.15)

    indices = rng.permutation(user_ids)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_data = ratings.loc[ratings["user_id"].isin(train_indices)]
    val_data = ratings.loc[ratings["user_id"].isin(val_indices)]
    test_data = ratings.loc[ratings["user_id"].isin(test_indices)]

    print(f"Train size: {len(train_data)}.")
    print(f"Validation size: {len(val_data)}.")
    print(f"Test size: {len(test_data)}.")

    train_data.to_csv(train_savepath, index=False)
    val_data.to_csv(val_savepath, index=False)
    test_data.to_csv(test_savepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data.")
    parser.add_argument("--input_directory", required=True, type=str, help="Directory containing the raw data.")
    parser.add_argument("--save_directory", required=True, type=str, help="Directory where processed data will be saved.")
    parser.add_argument("--create_train_val_test_split", action="store_true", help="Flag to indicate whether to create a train-validation split.")
    args = parser.parse_args()

    print("Processing raw data...")
    process_raw_data_goodreads(args.input_directory, args.save_directory)
    if args.create_train_val_test_split:
        create_train_val_test_split(
            os.path.join(args.save_directory, "ratings.csv"),
            os.path.join(args.save_directory, "train_ratings.csv"),
            os.path.join(args.save_directory, "val_ratings.csv"),
            os.path.join(args.save_directory, "test_ratings.csv")
        )
    print("The raw data has been successfully processed.")