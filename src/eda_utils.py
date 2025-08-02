import pandas as pd
from IPython.display import display, Image


def display_books_info(
    books: pd.DataFrame,
    include_cols: list[str] = []
) -> None:
    """
    Takes a books dataframe and displays each book's title, author, link and image.
    """
    for index, row in books.iterrows():
        print(f"Title: {row['title']}")
        print(f"Author: {row['author']}")
        print(f"Pages: {row['pages']}")
        for col in include_cols:
            print(f"{col.capitalize()}: {row[col]}")
        print(f"Link: https://www.goodreads.com/book/show/{row['bookId']}")
        display(Image(url=row['coverImg'], width=200, height=300))
        print("\n")
    