from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from functools import reduce
from ..utils import as_list

def gini(df, x, y, groupby = None, **kwargs):
    """
    Calculate gini for a Spark Dataframe
    df: Spark input dataframe
    x: list of columns to calculate gini for
    y: binary outcome variable
    groupby: optional list of columns to group by before calculating summary statistics
    """

    def _gini(df, x, y, groupby= None, **kwargs):
        return df\
            .select(x, y, *as_list(groupby))\
            .groupby(x, *as_list(groupby))\
            .agg(
                sum(col(y)).alias('b'),
                sum(1-col(y)).alias('g')
            )\
            .orderBy(as_list(groupby)+[desc(x)])\
            .withColumn('t_b', sum(col('b')).over(Window.partitionBy(*as_list(groupby))))\
            .withColumn('t_g', sum(col('g')).over(Window.partitionBy(*as_list(groupby))))\
            .withColumn('c_b', sum(col('b')).over(Window.partitionBy(*as_list(groupby)).orderBy(desc(x)).rowsBetween(Window.unboundedPreceding,0)) / col('t_b'))\
            .withColumn('c_g', sum(col('g')).over(Window.partitionBy(*as_list(groupby)).orderBy(desc(x)).rowsBetween(Window.unboundedPreceding,0)) / col('t_g'))\
            .withColumn('d_b', col('c_b')+lag(col('c_b')).over(Window.partitionBy(*as_list(groupby)).orderBy(desc(x))))\
            .withColumn('d_g', col('c_g')-lag(col('c_g')).over(Window.partitionBy(*as_list(groupby)).orderBy(desc(x))))\
            .withColumn('d_b', when(col('d_b').isNull(), 0).otherwise(col('d_b')))\
            .withColumn('d_g', when(col('d_g').isNull(), 0).otherwise(col('d_g')))\
            .withColumn('a', 0.5*col('d_b')*col('d_g'))\
            .groupby(*as_list(groupby))\
            .agg(
                sum(col('g')).alias('goods'),
                sum(col('b')).alias('bads'),
                sum(col('a')).alias('area_under_curve')
            )\
            .withColumn('gini', 2*(col('area_under_curve') - 0.5))\
            .withColumn('total', col('goods') + col('bads'))\
            .withColumn('rate', col('bads')/col('total'))\
            .select(as_list(groupby) + ['goods', 'bads', 'total', 'rate', 'area_under_curve', 'gini'])
    
    return reduce(DataFrame.union, [_gini(df=df, x=x, y=y, groupby=groupby, **kwargs).select(lit(x).alias('variable'), '*') for x in as_list(x)])
