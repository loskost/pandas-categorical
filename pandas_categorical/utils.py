from typing import Mapping, Union, List, Optional, Iterable, Sequence

import numpy as np
import pandas as pd
import pandas.core.common as com


def concat_categorical(dfs: Iterable[pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
    """
    Concatenate while preserving categorical columns.

    :param dfs: List of data frames
    :return: output dataset
    """
    dfs = list(com.not_none(*dfs))
    _union_categories(dfs)
    return pd.concat(dfs, *args, **kwargs)


def merge_categorical(
    left: pd.DataFrame,
    right: pd.DataFrame,
    remove_unused_categories: bool = False,
    **kwargs,
) -> pd.DataFrame:

    """
    Mergenate while preserving categorical columns.

    :param left: left dataframe
    :param right: right dataframe
    :param remove_unused_categories: bool, optional, remove unused categories
                                                     from categorical columns.
                                                     By default it is False.
    :return: output dataset
    """

    if all(col in kwargs for col in {"left_on", "right_on"}):
        left_on = kwargs["left_on"]
        right_on = kwargs["right_on"]
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]
        if not isinstance(left_on, list) or not isinstance(right_on, list):
            raise ValueError("'left_on' and 'right_on' must be list!")
        _union_left_right_categories(left, right, left_on, right_on)
    else:
        _union_categories([left, right])

    output = pd.merge(left, right, **kwargs)
    if remove_unused_categories:
        for col in set(output.select_dtypes(include="category").columns):
            output[col] = output[col].cat.remove_unused_categories()
    return output


def cat_astype(
    data: pd.DataFrame,
    cat_cols: Optional[Iterable[str]] = None,
    sub_dtypes: Optional[Mapping[str, Union[type, str]]] = None,
    ordered_cols: Optional[Iterable[str]] = None,
    remove_unused_categories: bool = False,
) -> None:
    """
    The function converts columns (cat_cols) to a categorical type
    with the specified types of the categories themselves (sub_dtypes).
    The conversion takes inplace.

    :param data: pd.DataFrame, input DataFrame.
    :param cat_cols: list or set, a list of columns that need to be converted.
                           If some column is not in data, it will be skipped.
    :param sub_dtypes: dict, dictionary with types for categories.
                             If the type is not specified for some column,
                             the category type will match the original column type.
    :param ordered_cols: set, set of columns whose categories are ordered.
    :param remove_unused_categories: bool, optional, remove unused categories
                                                     from categorical columns.
                                                     By default it is False.
    """
    if cat_cols is None:
        cat_cols = list()
    if sub_dtypes is None:
        sub_dtypes = dict()
    if ordered_cols is None:
        ordered_cols = set()
    already_cat = set(data.select_dtypes(include="category").columns)
    columns_to_transform = data.columns.intersection(list(cat_cols))
    for col in columns_to_transform:
        if col not in already_cat and col in ordered_cols:
            data[col] = pd.Categorical(
                data[col].to_numpy(),
                ordered=True,
                categories=np.sort(data[col].dropna().unique()),
            )
        if col not in already_cat and col not in ordered_cols:
            data[col] = data[col].astype("category")
        if (remove_unused_categories) and (
            data[col].dropna().unique().shape[0] != data[col].cat.categories.shape[0]
        ):
            data[col] = data[col].cat.remove_unused_categories()
        if col not in sub_dtypes:
            continue
        if data[col].cat.categories.dtype != sub_dtypes[col]:
            data[col] = data[col].cat.rename_categories(
                data[col].cat.categories.astype(sub_dtypes[col])
            )


def _union_categories(dfs: Sequence[pd.DataFrame]) -> None:

    cat_cols = set.intersection(
        *[set(df.select_dtypes(include="category").columns) for df in dfs]
    )
    for col in cat_cols:
        dtype = dfs[0][col].cat.categories.dtype
        uc = pd.Series(
            np.sort(
                list(set.union(*[set(df[col].cat.categories.to_list()) for df in dfs]))
            ),
            dtype=dtype,
        )
        for df in dfs:
            ordered_col = df[col].cat.ordered
            df[col] = pd.Categorical(df[col].values, categories=uc, ordered=ordered_col)


def _union_left_right_categories(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: List[str],
    right_on: List[str],
) -> None:

    left_cat_on = left.select_dtypes(include="category").columns.intersection(left_on)
    right_cat_on = right.select_dtypes(include="category").columns.intersection(
        right_on
    )
    for left_col, right_col in zip(left_cat_on, right_cat_on):
        left_col_ordered = left[left_col].cat.ordered
        right_col_ordered = right[right_col].cat.ordered
        uc = np.sort(
            list(
                set.union(
                    *[
                        set(df[col].cat.categories.to_list())
                        for df, col in zip((left, right), (left_col, right_col))
                    ]
                )
            )
        )
        left[left_col] = pd.Categorical(
            left[left_col].to_numpy(), categories=uc, ordered=left_col_ordered
        )
        right[right_col] = pd.Categorical(
            right[right_col].to_numpy(), categories=uc, ordered=right_col_ordered
        )
