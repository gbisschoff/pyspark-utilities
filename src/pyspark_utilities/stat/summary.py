from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from functools import reduce
from ..utils import as_list

def summary(df: DataFrame, columns: [str], groupby: [str] = None, percentiles:[float]=(0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99), **kwargs) -> DataFrame:
    """
    Calculate summary statistics for a Spark Dataframe
    df: Spark input dataframe
    columns: list of columns to calculate summary statistics for
    groupby: optional list of columns to group by before calculating summary statistics
    percentiles: list of percentiles to compute
    """
    def _summary(df: DataFrame, column: 'str', groupby: [str] = None, percentiles:[float]=(0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99), **kwargs) -> DataFrame:
        return df\
            .groupby(*as_list(groupby))\
            .agg(
                count(column).alias('n'),
                count(when(isnan(column) | isnull(column), column)).alias('missing'),
                sum(column).alias('sum'),
                avg(column).alias('average'),
                stddev(column).alias('stddev'),
                variance(column).alias('variance'),
                min(column).alias('min'),
                max(column).alias('max'),
                skewness(column).alias('skewness'),
                kurtosis(column).alias('kurtosis'),
                percentile_approx(column, percentiles).alias('percentile')
            ).select(['*'] + [col('percentile').getItem(i).alias('percentile_' + str(int(p * 100))) for i, p in enumerate(percentiles)])\
            .drop('percentile')

    return reduce(DataFrame.union, [_summary(df=df, column=c, groupby=groupby, percentiles=percentiles, **kwargs).select(lit(c).alias('groupby'), '*') for c in as_list(columns)])
