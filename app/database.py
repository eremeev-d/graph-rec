import io
import sqlite3

import numpy as np


class ItemDatabase:
    def __init__(self, db_path):
        sqlite3.register_converter("embedding", self._text_to_numpy_array)
        self._db_path = db_path

    @staticmethod
    def _text_to_numpy_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def _connect(self):
        return sqlite3.connect(
            self._db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def search_items(self, query, n_items=10):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(f"select item_id from items where title like '%{query}%'")
            rows = c.fetchall()[:n_items]
            return [row[0] for row in rows]

    def get_item(self, item_id):

        with self._connect() as conn:
            c = conn.cursor()
            c.row_factory = sqlite3.Row
            c.execute(f"select * from items where item_id like '{item_id}'")
            return c.fetchone()