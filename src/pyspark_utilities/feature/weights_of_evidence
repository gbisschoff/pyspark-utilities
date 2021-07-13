from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml.param.shared import HasInputCols, HasOutputCols, Param, Params, TypeConverters, HasLabelCol
# Available in PySpark >= 2.3.0 
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
from pyspark.ml.pipeline import Estimator, Model
from pyspark.sql.functions import col, create_map, lit, log, sum, when
from pyspark.sql.window import Window
from itertools import chain
from functools import reduce

class WeightsOfEvidence(Estimator, HasInputCols, HasOutputCols, HasLabelCol, DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, labelCol=None):
        super(WeightsOfEvidence, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, labelCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setLabelCol(self, value):
        return self._set(labelCol=value)

    def getLabelCol(self):
        return self.getOrDefault(self.labelCol)

    def _fit(self, dataframe):
        """Fit transformer."""
        def get_mapping(c):
            mapping_df = dataframe\
                .groupBy(c)\
                .agg(
                    sum(when(col(self.getLabelCol()) == 0, 1).otherwise(0)).alias("good"),
                    sum(when(col(self.getLabelCol()) == 1, 1).otherwise(0)).alias("bad")
                )\
                .withColumn('woe', log((col('good')/sum(col('good')).over(Window.partitionBy()))/(col('bad')/sum(col('bad')).over(Window.partitionBy()))))\
                .drop('good', 'bad')

            return reduce(lambda a, b: dict(a, **b), [{r[c]: r['woe']} for r in mapping_df.collect()])
        
        mappings = {c: get_mapping(c) for c in self.getInputCols()}
            
        return WeightsOfEvidenceModel(inputCols=self.getInputCols(), outputCols=self.getOutputCols(), mappings=mappings)


class WeightsOfEvidenceModel(Model, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable,):

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, mappings=None):
        """Initialize."""
        super(WeightsOfEvidenceModel, self).__init__()
        self.mappings = Param(self, "mappings", "WoE Mapping")
        self._setDefault(mappings={})
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None, mappings=None):
        """Get params."""
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setMappings(self, value):
        return self._set(mappings=value)

    def getMappings(self):
        return self.getOrDefault(self.mappings)

    def _transform(self, dataframe):
        def get_mapping_expr(mapping):          
            return create_map([lit(x) for x in chain(*mapping.items())])
        
        return dataframe.select(['*']+[(get_mapping_expr(self.getMappings()[i]).getItem(col(i))).alias(o) for i, o in zip(self.getInputCols(), self.getOutputCols())])       
