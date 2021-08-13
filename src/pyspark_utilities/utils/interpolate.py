import pyspark.sql.functions as F

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
        .join(df, on=[*as_list(groupby), index], how='left')
    )

def interpolate(df, index, groupby=None, fill_type='forward'):
    
    FILL_TYPE = {'forward':'ff', 'backward':'bf', 'interpolate':'in'}
    
    # define interpolation function
    def _interpolate(x, x_prev, x_next, y_prev, y_next, y):
        if x_prev == x_next:
            return y
        else:
            m = (y_next-y_prev)/(x_next-x_prev)
            y_interpol = y_prev + m * (x - x_prev)
            return y_interpol

    # convert function to udf
    interpol_udf = F.udf(_interpolate, FloatType())   
    
    #define forward and back fill windows
    window_ff = Window.partitionBy(*as_list(groupby))\
               .orderBy(index)\
               .rowsBetween(Window.unboundedPreceding, 0)
               
    window_bf = Window.partitionBy(*as_list(groupby))\
               .orderBy(index)\
               .rowsBetween(0, Window.unboundedFollowing)

   
    # create the series containing the filled values
    select_cols = [c for c in df.columns if not in groupby]
    for c in select_cols:        
        if fill_type in ('forward', 'interpolate'): df[c+'_ff'] = F.last(df[c], ignorenulls=True).over(window_ff)
        if fill_type in ('backward', 'interpolate'): df[c+'_bf'] = F.first(df[c], ignorenulls=True).over(window_bf)
        if fill_type is 'interpolate': df[c+'_in'] = interpol_udf(index, index+'_ff', index+'_bf', c+'_ff', c+'_bf', c)
            
    return df.select(*as_list(groupby), *[F.col(c+'_'+FILL_TYPE.get(fill_type)).alias(c) for c in select_cols])
  
