from typing import Iterable
from copy import deepcopy
from typing import Callable

import pandas as pd
import pytest

import pandas_categorical as pdc
from pandas.testing import assert_frame_equal
from pandas_categorical.utils import cat_astype, concat_categorical, merge_categorical

from tests.data_for_tests import (
    dfs_for_concat,
    left_right_dfs,
    dfs_for_merge_with_kwargs_and_answers,
    dfs_for_cat_astype_noreturn,
    dfs_for_cat_astype_with_answers,
)


def test_concat_categorical_output_type() -> None:
    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()

    res = concat_categorical([df_1, df_2])
    assert isinstance(res, pd.DataFrame)


@pytest.mark.parametrize("dfs, kwargs", deepcopy(dfs_for_concat))
def test_concat_categorical_standard_behavior(
    dfs: Iterable[pd.DataFrame], kwargs: dict
) -> None:
    answer = pd.concat(dfs, **kwargs)
    res = pdc.concat_categorical(dfs, **kwargs)

    assert_frame_equal(res, answer)


@pytest.mark.parametrize("dfs, kwargs", deepcopy(dfs_for_concat))
def test_concat_categorical_columns(dfs: Iterable[pd.DataFrame], kwargs: dict) -> None:
    dfs = [df.astype("category") for df in dfs]
    answer = pd.concat(dfs, **kwargs).astype("category")
    res = pdc.concat_categorical(dfs, **kwargs)

    assert_frame_equal(res, answer)


@pytest.mark.parametrize("dfs, kwargs", deepcopy(dfs_for_concat))
def test_concat_ordered_categorical_columns(
    dfs: Iterable[pd.DataFrame], kwargs: dict
) -> None:
    dfs = [df.astype(pd.CategoricalDtype(ordered=True)) for df in dfs]
    answer = pd.concat(dfs, **kwargs).astype(pd.CategoricalDtype(ordered=True))
    res = pdc.concat_categorical(dfs, **kwargs)

    assert_frame_equal(res, answer)


@pytest.mark.parametrize(
    "kwargs",
    [dict(on="a", how="left"), dict(on="a", how="right"), dict(on="a", how="outer")],
)
@pytest.mark.parametrize("left, right", left_right_dfs)
def test_merge_categorical_return_type(
    left: pd.DataFrame, right: pd.DataFrame, kwargs: dict
) -> None:
    res = merge_categorical(left, right, **kwargs)

    assert isinstance(res, pd.DataFrame)


@pytest.mark.parametrize(
    "kwargs",
    [dict(on="a", how="left"), dict(on="a", how="right"), dict(on="a", how="outer")],
)
@pytest.mark.parametrize("left, right", left_right_dfs)
def test_merge_categorical_standard_behavior(
    left: pd.DataFrame, right: pd.DataFrame, kwargs: dict
) -> None:
    answer = pd.merge(left, right, **kwargs)
    res = merge_categorical(left, right, **kwargs)

    assert_frame_equal(res, answer)


@pytest.mark.parametrize(
    "left, right, kwargs, extra_kwargs, make_answer",
    dfs_for_merge_with_kwargs_and_answers,
)
def test_merge_categorical_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    kwargs: dict,
    extra_kwargs: dict,
    make_answer: Callable[[pd.DataFrame, pd.DataFrame, dict], pd.DataFrame],
) -> None:
    res = merge_categorical(left, right, **kwargs, **extra_kwargs)
    answer = make_answer(left, right, kwargs)

    assert_frame_equal(res, answer)


@pytest.mark.parametrize("data, kwargs", dfs_for_cat_astype_noreturn)
def test_cat_astype_return_dtype(
    data: pd.DataFrame,
    kwargs: dict,
) -> None:
    assert cat_astype(data, **kwargs) is None


@pytest.mark.parametrize("data, kwargs, make_answer", dfs_for_cat_astype_with_answers)
def test_cat_astype(
    data: pd.DataFrame,
    kwargs: dict,
    make_answer: Callable[[pd.DataFrame, dict], pd.DataFrame],
) -> None:
    answer = make_answer(data, kwargs)
    cat_astype(data, **kwargs)

    assert_frame_equal(data, answer)


def test_replace_ordered_to_unordered() -> None:
    df = pd.DataFrame(
        {
            "Integer": [1, 4, 2, 5, 10],
            "Float": [1.0, 4.0, 2.0, 5.0, 10.0],
            "Bool": [True, False, False, True, False],
            "String": ["1", "4", "2", "5", "10"],
            "Object": [1, "4", 2, "5", 10],
            "id": range(5),
        }
    )
    cat_cols = ["Integer", "Float", "Bool", "String", "Object"]
    ordered_cols = {"Integer", "Bool", "String"}
    for col in cat_cols:
        df[col] = df[col].astype("category")
    for col in ordered_cols:
        df[col] = df[col].cat.as_ordered()

    new_ordered_cols = {"Float", "Bool", "Object"}
    answer = df.copy(deep=True)
    for col in new_ordered_cols.difference(ordered_cols):
        answer[col] = answer[col].cat.as_ordered()
    for col in ordered_cols.difference(new_ordered_cols):
        answer[col] = answer[col].cat.as_unordered()

    cat_astype(df, cat_cols=cat_cols, ordered_cols=new_ordered_cols)

    assert_frame_equal(df, answer)
