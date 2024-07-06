from typing import List, Tuple, Callable

import pandas as pd


dfs_for_concat = [
    (
        (pd.DataFrame({"int": [1, 2, 3]}), pd.DataFrame({"int": [4, 5, 6]})),
        dict(ignore_index=True),
    ),
    (
        (
            pd.DataFrame({"float": [1.0, 2.0, 3.0, float("nan")]}),
            pd.DataFrame({"float": [4.0, 5.0, 6.0]}),
        ),
        dict(ignore_index=True),
    ),
    (
        (
            pd.DataFrame({"object": ["1.0", "2.0", None]}),
            pd.DataFrame({"object": ["4.0", None, "6.0"]}),
        ),
        dict(ignore_index=True),
    ),
    (
        (
            pd.DataFrame({"string": ["1.0", "2.0", "3.0"]}, dtype="string"),
            pd.DataFrame({"string": ["4.0", "5.0", "6.0"]}, dtype="string"),
        ),
        dict(ignore_index=True),
    ),
    (
        (
            pd.DataFrame(
                {"date": pd.date_range("2000-01-01", periods=3)}, dtype="datetime64[ns]"
            ),
            pd.DataFrame(
                {"date": pd.date_range("2000-01-04", periods=3)}, dtype="datetime64[ns]"
            ),
        ),
        dict(ignore_index=True),
    ),
    (
        (
            pd.DataFrame({"bool": [True, True]}, dtype="bool"),
            pd.DataFrame({"bool": [False, False]}, dtype="bool"),
        ),
        dict(ignore_index=True),
    ),
]

left_right_dfs = [
    (
        pd.DataFrame({"a": ["key-1", "key-2"], "b": [1, 2]}),
        pd.DataFrame({"a": ["key-3", "key-2"], "c": [3, 4]}),
    ),
]

dfs_for_merge_with_kwargs_and_answers: List[
    Tuple[pd.DataFrame, pd.DataFrame, dict, dict, Callable]
] = [
    (
        pd.DataFrame({"a": ["key-1", "key-2"], "b": [1, 2]}).astype({"a": "category"}),
        pd.DataFrame({"a": ["key-3", "key-2"], "c": [3, 4]}).astype({"a": "category"}),
        dict(on="a", how="outer"),
        dict(),
        lambda left, right, kwargs: pd.merge(left, right, **kwargs).astype(
            {"a": "category"}
        ),
    ),
    (
        pd.DataFrame({"a": ["key-1", "key-2"], "b": [1, 2]}).astype("category"),
        pd.DataFrame({"a": ["key-3", "key-2"], "c": [3, 4]}).astype("category"),
        dict(on="a", how="outer"),
        dict(),
        lambda left, right, kwargs: pd.merge(left, right, **kwargs).astype(
            {"a": "category"}
        ),
    ),
    (
        pd.DataFrame({"al": ["key-1", "key-2"], "b": [1, 2]}).astype(
            {"al": "category"}
        ),
        pd.DataFrame({"ar": ["key-3", "key-2"], "c": [3, 4]}).astype(
            {"ar": "category"}
        ),
        dict(left_on="al", right_on="ar", how="outer"),
        dict(),
        lambda left, right, kwargs: pd.merge(left, right, **kwargs).astype(
            {"al": "category", "ar": "category"}
        ),
    ),
    (
        pd.DataFrame({"a": ["key-1", "key-2"], "b": [1, 2]}).astype(
            {"a": pd.CategoricalDtype(ordered=True)}
        ),
        pd.DataFrame({"a": ["key-3", "key-2"], "c": [3, 4]}).astype(
            {"a": pd.CategoricalDtype(ordered=True)}
        ),
        dict(on="a", how="outer"),
        dict(),
        lambda left, right, kwargs: pd.merge(left, right, **kwargs).astype(
            {"a": pd.CategoricalDtype(ordered=True)}
        ),
    ),
]

dfs_for_cat_astype_noreturn = [
    (
        pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]}),
        dict(cat_cols=["a", "b"]),
    ),
]

dfs_for_cat_astype_with_answers = [
    (
        pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]}),
        dict(cat_cols=["a", "b"]),
        lambda data, kwargs: data.astype("category"),
    ),
    (
        pd.DataFrame({"a": ["1", "2"], "b": ["1.0", "2.0"]}),
        dict(cat_cols=["a", "b"], sub_dtypes={"a": int, "b": float}),
        lambda data, kwargs: data.astype({"a": int, "b": float}).astype("category"),
    ),
    (
        pd.DataFrame({"a": ["1", "2"], "b": ["1.0", "2.0"]}).astype("category"),
        dict(cat_cols=["a", "b"], sub_dtypes={"a": int, "b": float}),
        lambda data, kwargs: data.astype({"a": int, "b": float}).astype("category"),
    ),
    (
        pd.DataFrame({"Integer": [1, 2], "Float": [10.0, 20.0]}),
        dict(
            cat_cols=["Integer", "Float"],
            sub_dtypes={"Integer": int, "Float": float},
            ordered_cols={"Integer", "Float"},
        ),
        lambda data, kwargs: data.astype(pd.CategoricalDtype(ordered=True)),
    ),
    (
        pd.DataFrame({"Integer": ["1", "2"], "Float": ["10.0", "20.0"]}),
        dict(
            cat_cols=["Integer", "Float"],
            sub_dtypes={"Integer": int, "Float": float},
            ordered_cols={"Integer", "Float"},
        ),
        lambda data, kwargs: data.astype({"Integer": int, "Float": float}).astype(
            pd.CategoricalDtype(ordered=True)
        ),
    ),
    (
        pd.DataFrame().assign(
            Integer=pd.Categorical(
                values=pd.Series([1, 2], dtype=int),
                categories=pd.Series([1, 2, 3], dtype=int),
                ordered=True,
            ),
            Float=pd.Categorical(
                values=pd.Series([10.0, 20.0], dtype=float),
                categories=pd.Series([10.0, 20.0, 30.0], dtype=float),
                ordered=True,
            ),
        ),
        dict(
            cat_cols=["Integer", "Float"],
            sub_dtypes={"Integer": int, "Float": float},
            ordered_cols={"Integer", "Float"},
            remove_unused_categories=True,
        ),
        lambda data, kwargs: pd.DataFrame(
            {"Integer": [1, 2], "Float": [10.0, 20.0]}
        ).astype(pd.CategoricalDtype(ordered=True)),
    ),
]
