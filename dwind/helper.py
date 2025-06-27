from __future__ import annotations

import numpy as np
import pandas as pd


def memory_downcaster(df):
    assert isinstance(df, pd.DataFrame) | isinstance(df, pd.Series)

    NAlist = []
    for col in df.select_dtypes(include=[np.number]).columns:
        IsInt = False
        mx = df[col].max()
        mn = df[col].min()

        # integer does not support na; fill na
        if not np.isfinite(df[col]).all():
            NAlist.append(col)
            df[col].fillna(mn - 1, inplace=True)

        # test if column can be converted to an integer
        asint = df[col].fillna(0).astype(np.int64)
        result = df[col] - asint
        result = result.sum()
        if result > -0.01 and result < 0.01:
            IsInt = True

        # make integer/unsigned integer datatypes
        if IsInt:
            try:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            except:  # noqa: E722
                df[col] = df[col].astype(np.float32)

        # make float datatypes 32 bit
        else:
            df[col] = df[col].astype(np.float32)

    return df


def interpolate_array(row, col_1, col_2, col_in, col_out):
    if row[col_in] != 0:
        interpolated = row[col_in] * (row[col_2] - row[col_1]) + row[col_1]
    else:
        interpolated = row[col_1]

    row[col_out] = interpolated

    return row


def scale_array_precision(df: pd.DataFrame, hourly_col: str, prec_offset_col: str):
    """Scales the precision of :py:attr:`hourly_col` by the :py:attr:`prec_offset_col`.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing :py:att:`hourly_col` and
            :py:att:`prec_offset_col`.
        hourly_col (str) The column to adjust the precision.
        prec_offset_col (str): The column for scaling the precison of :py:attr:`hourly_col`.

    Returns:
        pd.DataFrame: The input :py:attr:`df` with the precision of :py:attr:`hourly_col` scaled.
    """
    df[hourly_col] = (
        np.array(df[hourly_col].values.tolist(), dtype="float64")
        / df[prec_offset_col].values.reshape(-1, 1)
    ).tolist()
    return df


def scale_array_deprecision(df: pd.DataFrame, col: str | list[str]) -> pd.DataFrame:
    """Rounds the column(s) :py:attr:`col` to the nearest 2nd decimal and converts to NumPy's
    float32.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing :py:att:`col`.
        col (str | list[str]): The column(s) to have reduced precision.

    Returns:
        pd.DataFrame: The input :py:attr:`df` with the precision of :py:attr:`col` lowered.
    """
    df[col] = np.round(np.round(df[col], 2).astype(np.float32), 2)
    return df


def scale_array_sum(df: pd.DataFrame, hourly_col: str, scale_col: str) -> pd.DataFrame:
    """Scales the :py:attr:`hourly_col` by its sum and multiples by the :py:attr:`scale_col`.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the :py:attr:`hourly_col` and
            :py:attr:`scale_col`.
        hourly_col (str): The name of the column to be scaled whose values are lists.
        scale_col (str): The column to scale the :py:attr:`hourly_col`.

    Returns:
        pandas.DataFrame: The input dataframe, but with the values of the :py:attr:`hourly_col`
            scaled appropriately.
    """
    hourly_array = np.array(df[hourly_col].values.tolist())
    df[hourly_col] = (
        hourly_array / hourly_array.sum(axis=1).reshape(-1, 1) * df[scale_col].values.reshape(-1, 1)
    ).tolist()
    return df


def scale_array_multiplier(
    df: pd.DataFrame, hourly_col: str, multiplier_col: str, col_out: str
) -> pd.DataFrame:
    """Scales the :py:attr:hourly_col` values by the :py:attr:`multiplier_col`, and places it in
    the :py:attr:`col_out`.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing the :py:attr:`hourly_col` and
            :py:attr:`multiplier_col`.
        hourly_col (str): A column of hourly values as a list of floats in each cell.
        multiplier_col (str): The column used to scale the :py:attr:`hourly_col`.
        col_out (str): A new column that will contain the scaled data.

    Returns:
        pd.DataFrame: A new copy of the original data (:py:attr:`df`) containing the
            :py:attr:`col_out` column.
    """
    hourly_array = np.array(df[hourly_col].values.tolist())
    df[col_out] = (hourly_array * df[multiplier_col].values.reshape(-1, 1)).tolist()
    return df


def split_by_index(
    arr: pd.DataFrame | np.ndarray | pd.Series, n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a DataFrame, Series, or array like with np.array_split, but only return the start and
    stop indices, rather than chunks. For Pandas objects, this are equivalent to
    ``arr.iloc[start: end]`` and for NumPy: ``arr[start: end]``. Splits are done according
    to the 0th dimension.

    Args:
        arr(pd.DataFrame | pd.Series | np.ndarray): The array, data frame, or series to split.
        n_splits(:obj:`int`): The number of near equal or equal splits.

    Returns:
        tuple[np.ndarray, np.ndarray]
    """
    size = arr.shape[0]
    base = np.arange(n_splits)
    split_size = size // n_splits
    extra = size % n_splits

    starts = base * split_size
    ends = starts + split_size

    for i in range(extra):
        ends[i:] += 1
        starts[i + 1 :] += 1
    return starts, ends
