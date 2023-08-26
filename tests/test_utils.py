import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from pandas_categorical.utils import concat_categorical, merge_categorical, cat_astype


class TestConcatCategirical(unittest.TestCase):
    def test_output_type(self):
        df_1 = pd.DataFrame()
        df_2 = pd.DataFrame()

        res = concat_categorical([df_1, df_2], ignore_index=True)
        self.assertIsInstance(res, pd.DataFrame)

    def test_standard_behavior(self):
        df_1 = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": ["1", "2", "3"]})
        df_2 = pd.DataFrame({"a": [4, 5, 6], "b": [40, 50, 60], "c": ["4", "5", "6"]})

        answer = pd.concat([df_1, df_2], ignore_index=True)
        res = concat_categorical([df_1, df_2], ignore_index=True)

        assert_frame_equal(res, answer)

    def test_categorical_columns(self):
        df_1 = pd.DataFrame(
            {
                "Integer": [1, 2],
                "Float": [10.0, 20.0],
                "String": ["1", "2"],
                "Object": ["bla", float("nan")],
                "Date": pd.date_range("2000-01-01", periods=2),
                "Bool": [True, True],
            }
        )
        df_2 = pd.DataFrame(
            {
                "Integer": [3, 4],
                "Float": [30.0, 40.0],
                "String": ["3", "4"],
                "Object": ["bla_2", float("nan")],
                "Date": pd.date_range("2000-01-03", periods=2),
                "Bool": [False, False],
            }
        )
        df_3 = pd.DataFrame(
            {
                "Integer": [5, 6],
                "Float": [50.0, 60.0],
                "String": ["5", "6"],
                "Object": ["bla_3", float("nan")],
                "Date": pd.date_range("2000-01-05", periods=2),
                "Bool": [True, False],
            }
        )
        dtypes = {
            "Integer": int,
            "Float": float,
            "String": "string",
            "Object": object,
            "Date": "datetime64[ns]",
            "Bool": bool,
        }
        df_1 = df_1.astype(dtypes).astype("category")
        df_2 = df_2.astype(dtypes).astype("category")
        df_3 = df_3.astype(dtypes).astype("category")

        answer = pd.concat([df_1, df_2, df_3]).astype("category")
        res = concat_categorical([df_1, df_2, df_3])

        assert_series_equal(res.dtypes, answer.dtypes)
        assert_frame_equal(res, answer)

    def test_categorical_ordered_columns(self):

        df_1 = pd.DataFrame()
        df_1["Integer"] = pd.Categorical(
            values=pd.Series([1, 2], dtype=int),
            categories=pd.Series([1, 2], dtype=int),
            ordered=True,
        )
        df_1["Float"] = pd.Categorical(
            values=pd.Series([10.0, 20.0], dtype=float),
            categories=pd.Series([10.0, 20.0], dtype=float),
            ordered=True,
        )
        df_1["String"] = pd.Categorical(
            values=pd.Series(["1", "2"], dtype="string"),
            categories=pd.Series(["1", "2"], dtype="string"),
            ordered=True,
        )
        df_1["Object"] = pd.Categorical(
            values=pd.Series(["bla_1", float("nan")], dtype=object),
            categories=pd.Series(["bla_1"], dtype=object),
            ordered=True,
        )
        df_1["Date"] = pd.Categorical(
            values=pd.Series(
                pd.date_range("2000-01-01", periods=2), dtype="datetime64[ns]"
            ),
            categories=pd.Series(
                pd.date_range("2000-01-01", periods=2), dtype="datetime64[ns]"
            ),
            ordered=True,
        )
        df_1["Bool"] = pd.Categorical(
            values=pd.Series([True, True], dtype=bool),
            categories=pd.Series([True], dtype=bool),
            ordered=True,
        )

        df_2 = pd.DataFrame()
        df_2["Integer"] = pd.Categorical(
            values=pd.Series([3, 4], dtype=int),
            categories=pd.Series([3, 4], dtype=int),
            ordered=True,
        )
        df_2["Float"] = pd.Categorical(
            values=pd.Series([30.0, 40.0], dtype=float),
            categories=pd.Series([30.0, 40.0], dtype=float),
            ordered=True,
        )
        df_2["String"] = pd.Categorical(
            values=pd.Series(["3", "4"], dtype="string"),
            categories=pd.Series(["3", "4"], dtype="string"),
            ordered=True,
        )
        df_2["Object"] = pd.Categorical(
            values=pd.Series(["bla_2", float("nan")], dtype=object),
            categories=pd.Series(["bla_2"], dtype=object),
            ordered=True,
        )
        df_2["Date"] = pd.Categorical(
            values=pd.Series(
                pd.date_range("2000-01-03", periods=2), dtype="datetime64[ns]"
            ),
            categories=pd.Series(
                pd.date_range("2000-01-03", periods=2), dtype="datetime64[ns]"
            ),
            ordered=True,
        )
        df_2["Bool"] = pd.Categorical(
            values=pd.Series([False, False], dtype=bool),
            categories=pd.Series([False], dtype=bool),
            ordered=True,
        )

        answer = pd.concat([df_1, df_2], ignore_index=True)
        res = concat_categorical([df_1, df_2], ignore_index=True)

        answer["Integer"] = pd.Categorical(
            values=pd.Series([1, 2, 3, 4], dtype=int),
            categories=pd.Series([1, 2, 3, 4], dtype=int),
            ordered=True,
        )
        answer["Float"] = pd.Categorical(
            values=pd.Series([10.0, 20.0, 30.0, 40.0], dtype=float),
            categories=pd.Series([10.0, 20.0, 30.0, 40.0], dtype=float),
            ordered=True,
        )
        answer["String"] = pd.Categorical(
            values=pd.Series(["1", "2", "3", "4"], dtype="string"),
            categories=pd.Series(["1", "2", "3", "4"], dtype="string"),
            ordered=True,
        )
        answer["Object"] = pd.Categorical(
            values=pd.Series(
                ["bla_1", float("nan"), "bla_2", float("nan")], dtype=object
            ),
            categories=pd.Series(["bla_1", "bla_2"], dtype=object),
            ordered=True,
        )
        answer["Date"] = pd.Categorical(
            values=pd.Series(
                pd.date_range("2000-01-01", periods=4), dtype="datetime64[ns]"
            ),
            categories=pd.Series(
                pd.date_range("2000-01-01", periods=4), dtype="datetime64[ns]"
            ),
            ordered=True,
        )
        answer["Bool"] = pd.Categorical(
            values=pd.Series([True, True, False, False], dtype=bool),
            categories=pd.Series([False, True], dtype=bool),
            ordered=True,
        )

        assert_series_equal(res.dtypes, answer.dtypes)
        assert_frame_equal(res, answer)


class TestMergeCategirical(unittest.TestCase):
    @staticmethod
    def get_artificial_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        df_1 = pd.DataFrame({"a": ["key-1", "key-2"], "b": [1, 2]})
        df_2 = pd.DataFrame({"a": ["key-3", "key-2"], "c": [3, 4]})
        return df_1, df_2

    def test_output_type(self):
        df_1, df_2 = TestMergeCategirical.get_artificial_data()
        res = merge_categorical(df_1, df_2, on="a", how="left")
        self.assertIsInstance(res, pd.DataFrame)

    def test_standard_behavior(self):
        df_1, df_2 = TestMergeCategirical.get_artificial_data()

        answer = df_1.merge(df_2, on="a", how="left")
        res = merge_categorical(df_1, df_2, on="a", how="left")

        assert_frame_equal(res, answer)

    def test_categorical_columns_1(self):
        df_1, df_2 = TestMergeCategirical.get_artificial_data()

        df_1 = df_1.astype({"a": "category"})
        df_2 = df_2.astype({"a": "category"})

        answer = df_1.merge(df_2, on="a", how="outer").astype({"a": "category"})
        res = merge_categorical(df_1, df_2, on="a", how="outer")

        assert_series_equal(res.dtypes, answer.dtypes)
        assert_frame_equal(res, answer)

    def test_categorical_columns_2(self):
        df_1, df_2 = TestMergeCategirical.get_artificial_data()

        df_1 = df_1.astype("category")
        df_2 = df_2.astype("category")

        answer = df_1.merge(df_2, on="a", how="outer").astype({"a": "category"})
        res = merge_categorical(df_1, df_2, on="a", how="outer")

        assert_series_equal(res.dtypes, answer.dtypes)
        assert_frame_equal(res, answer)

    def test_categorical_columns_3(self):
        df_1, df_2 = TestMergeCategirical.get_artificial_data()
        df_1.rename(columns={"a": "al"}, inplace=True)
        df_2.rename(columns={"a": "ar"}, inplace=True)
        df_1 = df_1.astype({"al": "category"})
        df_2 = df_2.astype({"ar": "category"})

        answer = df_1.merge(df_2, left_on="al", right_on="ar", how="outer").astype(
            {"al": "category", "ar": "category"}
        )
        res = merge_categorical(
            df_1,
            df_2,
            left_on=["al"],
            right_on=["ar"],
            how="outer",
            remove_unused_categories=True,
        )

        assert_series_equal(res.dtypes, answer.dtypes)
        assert_frame_equal(res, answer)

    def test_categorical_columns_4(self):
        df_1, df_2 = TestMergeCategirical.get_artificial_data()

        df_1["a"] = pd.Categorical(
            df_1["a"], categories=np.sort(df_1["a"].dropna().unique()), ordered=True
        )
        df_2["a"] = pd.Categorical(
            df_2["a"], categories=np.sort(df_2["a"].dropna().unique()), ordered=True
        )

        answer = pd.merge(df_1, df_2, on="a", how="outer")
        answer["a"] = pd.Categorical(
            answer["a"], categories=np.sort(answer["a"].dropna().unique()), ordered=True
        )
        res = merge_categorical(df_1, df_2, on="a", how="outer")
        assert_series_equal(res.dtypes, answer.dtypes)
        assert_frame_equal(res, answer)


class TestCatAstype(unittest.TestCase):
    def test_no_return(self):
        self.assertIs(cat_astype(pd.DataFrame()), None)

    def test_unordered_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})

        answer = df.copy(deep=True)
        answer = answer.astype("category")

        cat_astype(df, cat_cols=["a", "b"])

        assert_series_equal(df.dtypes, answer.dtypes)
        assert_frame_equal(df, answer)

    def test_change_subtypes_1(self):
        df = pd.DataFrame({"a": ["1", "2"], "b": ["1.0", "2.0"]})

        answer = df.copy(deep=True)
        answer = answer.astype({"a": int, "b": float}).astype("category")

        cat_astype(df, cat_cols=["a", "b"], sub_dtypes={"a": int, "b": float})

        assert_series_equal(df.dtypes, answer.dtypes)
        assert_frame_equal(df, answer)

    def test_change_subtypes_2(self):
        df = pd.DataFrame({"a": ["1", "2"], "b": ["1.0", "2.0"]})
        df = df.astype("category")

        answer = df.copy(deep=True)
        answer = answer.astype({"a": int, "b": float}).astype("category")

        cat_astype(df, cat_cols=["a", "b"], sub_dtypes={"a": int, "b": float})

        assert_series_equal(df.dtypes, answer.dtypes)
        assert_frame_equal(df, answer)

    def test_ordered_categories(self):
        df = pd.DataFrame({"Integer": [1, 2], "Float": [10.0, 20.0]})

        answer = df.copy(deep=True)
        answer["Integer"] = pd.Categorical(
            values=pd.Series([1, 2], dtype=int),
            categories=pd.Series([1, 2], dtype=int),
            ordered=True,
        )
        answer["Float"] = pd.Categorical(
            values=pd.Series([10.0, 20.0], dtype=float),
            categories=pd.Series([10.0, 20.0], dtype=float),
            ordered=True,
        )
        cat_astype(
            df,
            cat_cols=["Integer", "Float"],
            sub_dtypes={"Integer": int, "Float": float},
            ordered_cols={"Integer", "Float"},
        )

        assert_series_equal(df.dtypes, answer.dtypes)
        assert_frame_equal(df, answer)

    def test_ordered_categories_with_subtypes(self):
        df = pd.DataFrame({"Integer": ["1", "2"], "Float": ["10.0", "20.0"]})

        answer = df.copy(deep=True)
        answer["Integer"] = pd.Categorical(
            values=pd.Series([1, 2], dtype=int),
            categories=pd.Series([1, 2], dtype=int),
            ordered=True,
        )
        answer["Float"] = pd.Categorical(
            values=pd.Series([10.0, 20.0], dtype=float),
            categories=pd.Series([10.0, 20.0], dtype=float),
            ordered=True,
        )
        cat_astype(
            df,
            cat_cols=["Integer", "Float"],
            sub_dtypes={"Integer": int, "Float": float},
            ordered_cols={"Integer", "Float"},
        )

        assert_series_equal(df.dtypes, answer.dtypes)
        assert_frame_equal(df, answer)

    def test_remove_unused_categories(self):
        df = pd.DataFrame()
        df["Integer"] = pd.Categorical(
            values=pd.Series([1, 2], dtype=int),
            categories=pd.Series([1, 2, 3], dtype=int),
            ordered=True,
        )
        df["Float"] = pd.Categorical(
            values=pd.Series([10.0, 20.0], dtype=float),
            categories=pd.Series([10.0, 20.0, 30.0], dtype=float),
            ordered=True,
        )

        answer = df.copy(deep=True)
        answer["Integer"] = pd.Categorical(
            values=pd.Series([1, 2], dtype=int),
            categories=pd.Series([1, 2], dtype=int),
            ordered=True,
        )
        answer["Float"] = pd.Categorical(
            values=pd.Series([10.0, 20.0], dtype=float),
            categories=pd.Series([10.0, 20.0], dtype=float),
            ordered=True,
        )
        cat_astype(
            df,
            cat_cols=["Integer", "Float"],
            sub_dtypes={"Integer": int, "Float": float},
            ordered_cols={"Integer", "Float"},
            remove_unused_categories=True,
        )

        assert_series_equal(df.dtypes, answer.dtypes)
        assert_frame_equal(df, answer)
