import sqlite3
import sys

import streamlit as st

from app.database import ItemDatabase
from app.recommendations import RecommenderSystem


def show_item(item_id):
    item = st.session_state["db"].get_item(item_id)
    title = item["title"]
    with st.container(border=True):
        st.write(f"**{title}**")
        st.write(item["description"])
        if st.button("Recommend similar items", key=item["item_id"]):
            st.session_state["recommendation_query"] = item["item_id"]
            st.session_state["search_query"] = None  # reset
            st.rerun()


def main():
    st.title("Graph-based RecSys")

    if "db" not in st.session_state:
        st.session_state["db"] = ItemDatabase(
            db_path="/data/items.db")
    if "recsys" not in st.session_state:
        st.session_state["recsys"] = RecommenderSystem(
            faiss_index_path="/data/index.faiss",
            db_path="/data/items.db")

    if "search_query" not in st.session_state:
        st.session_state["search_query"] = None
    if "recommendation_query" not in st.session_state:
        st.session_state["recommendation_query"] = None

    search_query = st.text_input("Enter item name", st.session_state["search_query"])

    if st.button("Search"):
        st.session_state["search_query"] = search_query
        st.session_state["recommendation_query"] = None  # reset

    if st.session_state["recommendation_query"] is not None:
        query = st.session_state["recommendation_query"]
        base_item_title = st.session_state["db"].get_item(query)["title"]
        st.subheader(f'Recommendation Results for "{base_item_title}"')
        results = st.session_state["recsys"].recommend_items(query)
        for item_id in results:
            show_item(item_id)

    elif st.session_state["search_query"] is not None:
        query = st.session_state["search_query"]
        st.subheader(f'Search Results for "{query}"')
        results = st.session_state["db"].search_items(query)
        for item_id in results:
            show_item(item_id)

if __name__ == "__main__":
    main()