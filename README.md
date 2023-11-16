[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

The package contains just a few features that make using pandas categorical types easier to use.
The main purpose of using categorical types is to reduce RAM consumption when working with large datasets. Experience shows that on average, there is a decrease of 2 times (for datasets of several GB, this is very significant). The full justification of the reasons and examples will be given below.
# Quickstart

```
pip install pandas-categorical
```

```python
import pandas as pd
import pandas-categorical as pdc
```

```python
df.astype('category')     ->     pdc.cat_astype(df, ...)
pd.concat()               ->     pdc.concat_categorical()
pd.merge()                ->     pdc.merge_categorical()
df.groupby(...)           ->     df.groupby(..., observed=True)
```
## cat_astype

```python
df = pd.read_csv("path_to_dataframe.csv")

SUB_DTYPES = {
	'cat_col_with_int_values': int,
	'cat_col_with_string_values': 'string',
	'ordered_cat_col_with_bool_values': bool,
}
pdc.cat_astype(
	data=df,
	cat_cols=SUB_DTYPES.keys(),
	sub_dtypes=SUB_DTYPES,
	ordered_cols=['ordered_cat_col_with_bool_values']
)
```

## concat_categorical

```python
df_1 = ...  # dataset with some categorical columns
df_2 = ...  # dataset with some categorical columns (categories are not equals)

df_res = pdc.concat_categorical((df_1, df_2), axis=0, ignore_index=True)
```

## merge_categorical

```python
df_1 = ...  # dataset with some categorical columns
df_2 = ...  # dataset with some categorical columns (categories are not equals)

df_res = pdc.merge_categorical(df_1, df_2, on=['cat_col_1', 'cat_col_2'])
```

# A bit of theory

The advantages are discussed in detail in the articles [here](https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a), [here](https://towardsdatascience.com/pandas-groupby-aggregate-transform-filter-c95ba3444bbb) and [here](https://pandas.pydata.org/docs/user_guide/categorical.html).

The categorical type implies the presence of a certain set of unique values in this column, which are often repeated. By reducing the copying of identical values, it is possible to reduce the size of the column (the larger the dataset, the more likely repetitions are). By default, categories (unique values) have no order. That is, they are not comparable to each other. It is possible to make them ordered.

Pandas already has everything for this (for example, `.astype(’category’)`). However, standard methods, in my opinion, require a high entry threshold and therefore are rarely used.

Let's try to outline a number of problems and ways to solve them.

## 1. Categorical types are easy to lose

Suppose you want to connect two datasets into one using `pd.concat(..., axis=0)'. Datasets contain columns with categorical types.
If the column categories of the source datasets are different, then `pandas` it does not combine multiple values, but simply resets their default type (for example, `object`, `int`, ...).
In other words,
$$\textcolor{red}{category1} + \textcolor{green}{category2} = object$$
$$\textcolor{red}{category1} + \textcolor{red}{category1} = \textcolor{red}{category1}$$
But we would like to observe a different behavior:
$$\textcolor{red}{category1} + \textcolor{green}{category2} = \textcolor{blue}{category3}$$
$$(\textcolor{blue}{category3} = \textcolor{red}{category1} \cup \textcolor{green}{category2})$$
As a result, you need to monitor the reduction of categories before actions such as `merge` or `concat'.
## 2.  Categories type control

When you do a type conversion
```python
df['col_name'] = df['col_name'].astype('category')
```
the type of categories is equal to the type of the source column.
But, if you want to change the type of categories, you probably want to write something like
```python
df['col_name'] = df['col_name'].astype('some_new_type').astype('category')

```
That is, you will temporarily lose the categorical type (respectively, and the advantages of memory).
By the way, the usual way of control
```python
df.dtypes
```
does not display information about the type of categories themselves. You will only see only `category` next to the desired column.

## 3. Unused values

Suppose you have filtered the dataset. At the same time, the actual set of values of categorical columns could decrease, but the data type will not be changed.
This can negatively affect, for example, when working with `groupby` on such a column. As a result, grouping will also occur by unused categories. To prevent this from happening, you need to specify the `observed=True` parameter.
For example,
```python
df.groupby(['cat_col_1', 'cat_col_2'], observed=True).agg('mean')
```

## 4. Ordered categories

There is no simple method to create ordered categorical columns.
In addition to the name, you need to explicitly specify the categories themselves in the right order.
To simplify, it is suggested to always use the same sort order.
For example,
```python
df['ordered_col'] = pd.Categorical(
	df['ordered_col'],
	categories=np.sort(data['ordered_col'].dropna().unique()),
	ordered=True,
)
```
Note: Categories cannot contain `NaN` values, so `.dropna()` is required.

## 5. Minimum copying

To process large datasets, you need to minimize the creation of copies of even its parts. Therefore, the functions from this package do the transformations in place.


## 6. Data storage in parquet format

When using `pd.to_parquet(path, engine='pyarrow')` and `pd.read_parque(path, engine='pyarrow')`categorical types of some columns can be reset to normal. To solve this problem, you can use `engine='fast parquet'. 

Note 1: `fastparquet` usually runs a little slower than `pyarrow'.

Note 2: `pyarrow` and `fastparquet` cannot be used together (for example, save by one and read by the other). This can lead to data loss.

```python
import pandas as pd


df = pd.DataFrame(
	{
		"Date": pd.date_range('2023-01-01', periods=10),
		"Object": ["a"]*5+["b"]+["c"]*4,
		"Int": [1, 1, 1, 2, 3, 1, 2, 4, 3, 2],
		"Float": [1.1]*5+[2.2]*5,
	}
)

print(df.dtypes)
df = df.astype('category')
print(df.dtypes)
df.to_parquet('test.parquet', engine='pyarrow')
df = pd.read_parquet('test.parquet', engine='pyarrow')
print(df.dtypes)
```
Output:
```
Date      datetime64[ns]
Object    object
Int       int64
Float     float64
dtype: object

Date      category
Object    category
Int       category
Float     category
dtype: object

Date      datetime64[ns]
Object    category
Int       int64
Float     float64
dtype: object
```

# Examples

[Jupiter notebook with examples](https://www.kaggle.com/code/loskost/problems-of-pandas-categorical-dtypes) of problems is posted on kaggle. A copy can be found in the `examples/` folder.
# Remarks

1. Processing of categorical indexes has not yet been implemented.
2. In the future, the function `pdc.join_categorical()` will appear.
3. The `cat_astype` function was designed so that the type information could be redundant (for example, it is specified for all possible column names in the project at once). In the future, it will be possible to set default values for this function.
# Links

1. [Official pandas documentation](https://pandas.pydata.org/docs/user_guide/categorical.html).
2. https://towardsdatascience.com/staying-sane-while-adopting-pandas-categorical-datatypes-78dbd19dcd8a
3. https://towardsdatascience.com/pandas-groupby-aggregate-transform-filter-c95ba3444bbb
4. [The source of the idea](https://stackoverflow.com/questions/45639350/retaining-categorical-dtype-upon-dataframe-concatenation) that I wanted to develop.
