import pyspark.sql.functions as F
import pyspark.sql.types as Types
from pyspark.sql.window import Window

def as_list(x):
    if isinstance(x, list): return x
    if x is None: return []
    else: return [x]
    

def resample(df, index, groupby=None, step=1, interval='month'):
    return (
        df
        .groupby(*as_list(groupby))
        .agg(
            F.min(F.col(index)).cast('date').alias('start'),
            F.max(F.col(index)).cast('date').alias('end'),
        )
        .withColumn(index, F.explode(F.expr('sequence(start, end, interval {step} {interval})'.format(step=step, interval=interval))))
        .drop('start', 'end')
        .join(df, on=[*as_list(groupby), index], how='left')
    )

def interpolate(df, index, cols, groupby=None, fill_type='forward'):

    # define interpolation function
    def _interpolate(x, x_prev, x_next, y_prev, y_next, y):
        if x_prev == x_next:
            return y
        else:
            m = (y_next-y_prev)/(x_next-x_prev)
            y_interpol = y_prev + m * (x - x_prev)
            return y_interpol

    # convert function to udf
    interpol_udf = F.udf(_interpolate, Types.FloatType())
    
    #define forward and back fill windows
    window_ff = Window.partitionBy(*as_list(groupby))\
               .orderBy(index)\
               .rowsBetween(Window.unboundedPreceding, 0)
               
    window_bf = Window.partitionBy(*as_list(groupby))\
               .orderBy(index)\
               .rowsBetween(0, Window.unboundedFollowing)

   
    # create the series containing the filled values
    for c in as_list(cols):
        if fill_type == 'forward':
            df = df.withColumn(c+'_ff', F.last(df[c], ignorenulls=True).over(window_ff))\
                .drop(c).withColumnRenamed(c + '_ff', c)
        elif fill_type == 'backward':
            df = df.withColumn(c+'_bf', F.first(df[c], ignorenulls=True).over(window_bf))\
                .drop(c).withColumnRenamed(c+'_bf', c)
        elif fill_type == 'interpolate':
            df = df.withColumn(c + '_ff', F.last(df[c], ignorenulls=True).over(window_ff))
            df = df.withColumn(c + '_bf', F.first(df[c], ignorenulls=True).over(window_bf))
            df = df.withColumn('_nobs', F.row_number().over(window_ff))
            df = df.withColumn('_nobs_ff', F.last(F.when(df[c].isNotNull(),df['_nobs']), ignorenulls=True).over(window_ff))
            df = df.withColumn('_nobs_bf', F.first(F.when(df[c].isNotNull(),df['_nobs']), ignorenulls=True).over(window_bf))

            df = df\
                .withColumn(c+'_in',
                    interpol_udf(
                        '_nobs',
                        '_nobs_ff',
                        '_nobs_bf',
                        c+'_ff',
                        c+'_bf',
                        c
                    )
                )\
                .drop('_nobs', '_nobs_ff', '_nobs_bf', c, c+'_ff', c+'_bf')\
                .withColumnRenamed(c+'_in', c)

    return df
