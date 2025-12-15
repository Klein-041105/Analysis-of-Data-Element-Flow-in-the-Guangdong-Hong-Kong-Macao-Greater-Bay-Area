"""数据验证工具模块。

提供一组轻量级的验证函数，用于在数据处理管道中进行类型和值的检查。
所有验证函数在条件不满足时抛出异常，支持链式调用。

主要功能：
- 空值检查
- 数据类型验证
- 维度和长度检查
- 数据范围验证

使用示例：
    >>> from validators import ensure_not_empty, ensure_numeric_array
    >>> data = [1, 2, 3]
    >>> arr = ensure_numeric_array(data, "input_data")
    >>> ensure_same_length(arr, labels, "features", "targets")
"""

from typing import Any, Iterable
import numpy as np
import pandas as pd


def ensure_not_empty(obj: Any, name: str = "object"):
    """确保对象非空。
    
    验证对象不为None且长度大于0（对于可迭代对象）。
    
    Args:
        obj: 要验证的对象，可以是None、数组、列表、DataFrame、Series等
        name: 对象的名称，用于错误消息中的上下文
        
    Raises:
        ValueError: 当obj为None或为空集合时
        
    Examples:
        >>> ensure_not_empty([1, 2, 3], "data")  # 通过
        >>> ensure_not_empty([], "data")  # 抛出ValueError
        >>> ensure_not_empty(None, "data")  # 抛出ValueError
    """
    if obj is None:
        raise ValueError(f"{name} must not be None.")

    if isinstance(obj, (np.ndarray, list, pd.DataFrame, pd.Series)) and len(obj) == 0:
        raise ValueError(f"{name} must not be empty.")


def ensure_numeric_array(arr: Any, name: str = "array") -> np.ndarray:
    """确保输入为数值数组。
    
    验证输入对象非空且包含数值类型数据。
    自动将列表等可迭代对象转换为numpy数组。
    
    Args:
        arr: 输入对象，可以是列表、数组、Series等
        name: 对象的名称，用于错误消息中的上下文
        
    Returns:
        numpy数组，dtype为数值类型（float, int等）
        
    Raises:
        ValueError: 当arr为空时（通过ensure_not_empty）
        TypeError: 当arr包含非数值数据时
        
    Examples:
        >>> arr = ensure_numeric_array([1, 2, 3], "values")
        >>> print(type(arr))  # <class 'numpy.ndarray'>
        >>> ensure_numeric_array(['a', 'b'], "data")  # 抛出TypeError
    """
    ensure_not_empty(arr, name)

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values.")

    return arr


def ensure_same_length(a: Iterable, b: Iterable, name_a="a", name_b="b"):
    """确保两个可迭代对象长度相同。
    
    用于验证配对的数据（如特征和标签）长度一致。
    
    Args:
        a: 第一个可迭代对象
        b: 第二个可迭代对象
        name_a: 第一个对象的名称，用于错误消息
        name_b: 第二个对象的名称，用于错误消息
        
    Raises:
        ValueError: 当len(a) != len(b)时
        
    Examples:
        >>> X = [1, 2, 3]
        >>> y = [0, 1, 0]
        >>> ensure_same_length(X, y, "features", "labels")  # 通过
        >>> y_wrong = [0, 1]
        >>> ensure_same_length(X, y_wrong, "features", "labels")  # 抛出ValueError
    """
    if len(a) != len(b):
        raise ValueError(f"{name_a} and {name_b} must have the same length.")


def ensure_dataframe(df: Any, name="dataframe") -> pd.DataFrame:
    """确保输入为非空的pandas DataFrame。
    
    综合验证：检查类型为DataFrame且非空。
    
    Args:
        df: 要验证的对象
        name: 对象的名称，用于错误消息中的上下文
        
    Returns:
        验证通过的DataFrame对象
        
    Raises:
        TypeError: 当df不是pandas DataFrame时
        ValueError: 当df为空时（通过ensure_not_empty）
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> validated = ensure_dataframe(df, "data")
        >>> ensure_dataframe([1, 2], "data")  # 抛出TypeError
        >>> ensure_dataframe(pd.DataFrame(), "data")  # 抛出ValueError
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    ensure_not_empty(df, name)
    return df


def ensure_series(s: Any, name: str = "series") -> pd.Series:
    """确保输入为非空的pandas Series。
    
    综合验证：检查类型为Series且非空。
    
    Args:
        s: 要验证的对象
        name: 对象的名称，用于错误消息中的上下文
        
    Returns:
        验证通过的Series对象
        
    Raises:
        TypeError: 当s不是pandas Series时
        ValueError: 当s为空时
        
    Examples:
        >>> s = pd.Series([1, 2, 3])
        >>> validated = ensure_series(s, "target")
        >>> ensure_series([1, 2], "target")  # 抛出TypeError
    """
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    ensure_not_empty(s, name)
    return s


def ensure_no_missing_values(df: pd.DataFrame, name: str = "dataframe") -> pd.DataFrame:
    """确保DataFrame中没有缺失值。
    
    检查并验证数据框中不存在NaN或None值。
    对数据质量验证很重要。
    
    Args:
        df: 要验证的DataFrame
        name: 对象的名称，用于错误消息中的上下文
        
    Returns:
        验证通过的DataFrame对象
        
    Raises:
        ValueError: 当存在缺失值时，给出缺失值的详细信息
        
    Examples:
        >>> df_clean = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> ensure_no_missing_values(df_clean, "data")
        >>> df_missing = pd.DataFrame({'a': [1, None], 'b': [3, 4]})
        >>> ensure_no_missing_values(df_missing)  # 抛出ValueError
    """
    if df.isnull().any().any():
        missing_info = df.isnull().sum()
        raise ValueError(
            f"{name} contains missing values:\n{missing_info[missing_info > 0]}"
        )
    return df


def ensure_numeric_columns(df: pd.DataFrame, columns: list, name: str = "dataframe") -> pd.DataFrame:
    """确保DataFrame中指定的列为数值类型。
    
    验证指定列的数据类型都是数值（int、float等）。
    
    Args:
        df: 要验证的DataFrame
        columns: 要检查的列名列表
        name: 对象的名称，用于错误消息中的上下文
        
    Returns:
        验证通过的DataFrame对象
        
    Raises:
        ValueError: 当指定的列不存在时
        TypeError: 当指定的列包含非数值数据时
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        >>> ensure_numeric_columns(df, ['a'], "features")  # 通过
        >>> ensure_numeric_columns(df, ['b'], "features")  # 抛出TypeError
        >>> ensure_numeric_columns(df, ['c'], "features")  # 抛出ValueError
    """
    # 检查列是否存在
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{name}: columns {missing_cols} not found in DataFrame"
        )
    
    # 检查列的数据类型
    non_numeric = []
    for col in columns:
        if not np.issubdtype(df[col].dtype, np.number):
            non_numeric.append(col)
    
    if non_numeric:
        raise TypeError(
            f"{name}: columns {non_numeric} must be numeric"
        )
    
    return df


def ensure_column_values_in_range(
    df: pd.DataFrame, column: str, min_val: float = None, max_val: float = None, name: str = "dataframe"
) -> pd.DataFrame:
    """确保DataFrame中指定列的值在指定范围内。
    
    验证列中的所有数值都在[min_val, max_val]范围内。
    
    Args:
        df: 要验证的DataFrame
        column: 列名
        min_val: 最小值（包含），None表示无下限
        max_val: 最大值（包含），None表示无上限
        name: 对象的名称，用于错误消息中的上下文
        
    Returns:
        验证通过的DataFrame对象
        
    Raises:
        ValueError: 当列不存在或值超出范围时
        
    Examples:
        >>> df = pd.DataFrame({'score': [0.1, 0.5, 0.9]})
        >>> ensure_column_values_in_range(df, 'score', 0, 1)  # 通过
        >>> df_bad = pd.DataFrame({'score': [0.1, 1.5, 0.9]})
        >>> ensure_column_values_in_range(df_bad, 'score', 0, 1)  # 抛出ValueError
    """
    if column not in df.columns:
        raise ValueError(f"{name}: column '{column}' not found")
    
    col_data = df[column]
    
    if min_val is not None and (col_data < min_val).any():
        raise ValueError(
            f"{name}: column '{column}' has values below minimum {min_val}"
        )
    
    if max_val is not None and (col_data > max_val).any():
        raise ValueError(
            f"{name}: column '{column}' has values above maximum {max_val}"
        )
    
    return df


def ensure_unique_index(df: pd.DataFrame, name: str = "dataframe") -> pd.DataFrame:
    """确保DataFrame的索引是唯一的。
    
    验证索引中没有重复值。
    
    Args:
        df: 要验证的DataFrame
        name: 对象的名称，用于错误消息中的上下文
        
    Returns:
        验证通过的DataFrame对象
        
    Raises:
        ValueError: 当索引中存在重复值时
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})  # 默认索引是唯一的
        >>> ensure_unique_index(df)  # 通过
        >>> df_dup = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 0, 1])
        >>> ensure_unique_index(df_dup)  # 抛出ValueError
    """
    if not df.index.is_unique:
        duplicates = df.index[df.index.duplicated()].unique()
        raise ValueError(
            f"{name}: index contains duplicate values: {duplicates.tolist()}"
        )
    return df
