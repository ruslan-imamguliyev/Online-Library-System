import pandas as pd
from IPython.display import display, Image
import ast


def display_books_info(books: pd.DataFrame) -> None:
    """
    Takes a `books` dataframe and displays each book's title, author, link and image.
    """
    for index, row in books.iterrows():
        print(f"Title: {row['title']}")
        if type(row['author']) == list:
            print(f"Author: {", ".join(row['author'])}")
        else:
            print(f"Author: {", ".join(ast.literal_eval(row['author']))}")
        print(f"Pages: {row['pages']}")
        print(f"Link: {row['link']}")
        display(Image(url=row['img'], width=200, height=300))
        print("\n")
    