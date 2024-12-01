"""All utilities related to string pre-processing"""
import pandas as pd


def remove_unwanted_characters(
    df: pd.DataFrame, col: str, replaceable_characters: str, new_characters: str = ""
):
    """
    Replace unwanted characters from the value

    Args:
        df (pd.DataFrame): dataframe
        col (str): name of the column
        replaceable_characters (str): set of substring to replace
        new_characters (str, optional): new substring to be replaced, Defaults to "".
    """
    df[col] = df[col].str.replace(replaceable_characters, new_characters)
