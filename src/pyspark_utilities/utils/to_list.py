import pyspark.sql.types as Types
from pyspark.sql.functions import udf


def to_list(col):
    def _to_list(v):
        return v.toArray().tolist()
    return udf(_to_list, Types.ArrayType(Types.DoubleType()))(col)
