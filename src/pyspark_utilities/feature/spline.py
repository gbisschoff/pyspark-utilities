import numpy as np
from pyspark.sql import DataFrame
from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, TypeConverters
# Available in PySpark >= 2.3.0
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
import pyspark.sql.types as Types
from pyspark.sql.functions import udf, col
from pyspark.ml.linalg import Vectors, VectorUDT
from .shared import HasOrder, HasIntercept, HasKnots


class Spline(Transformer, HasInputCol, HasOutputCol, HasOrder, HasIntercept, HasKnots, DefaultParamsReadable,
                       DefaultParamsWritable):


    @keyword_only
    def __init__(self, inputCol: str=None, outputCol: str=None, order: int=None, intercept: bool=None, knots: [float]=None):
        super(Spline, self).__init__()
        self._setDefault(order=3, intercept=False, knots=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol: str=None, outputCol: str=None, order: int=None, intercept: bool=None, knots: [float]=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        knots = self.getKnots()
        order = self.getOrder()
        start = 0 if self.getIntercept() else 1

        def f(x):
            return Vectors.dense(
                np.hstack((x ** np.arange(start, order), np.maximum(np.subtract(x, knots), 0) ** order)).astype(
                    float).tolist())

        return dataset.withColumn(self.getOutputCol(), udf(f, VectorUDT())(col(self.getInputCol())))
