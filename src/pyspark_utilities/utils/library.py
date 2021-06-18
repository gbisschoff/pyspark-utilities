from abc import ABC
import os.path
import re
from pyspark.sql import DataFrame, SparkSession


class Library(ABC):

    def __init__(self):
        pass

    def list(self):
        raise NotImplementedError()

    def read(self, name):
        raise NotImplementedError()

    def write(self, df, name):
        raise NotImplementedError()

    def delete(self, name):
        raise NotImplementedError()

    def update(self, df, name):
        raise NotImplementedError()


class MemoryLibrary(Library):

    def __init__(self):
        super(MemoryLibrary, self).__init__()
        self.MEMLIB = dict()

    def read(self, name):
        return self.MEMLIB.get(name)

    def write(self, df, name):
        self.MEMLIB[name] = df
        return self.read(name)

    def list(self):
        return self.MEMLIB.items()

    def delete(self, name):
        del self.MEMLIB[name]

    def update(self, df, name):
        self.delete(name)
        return self.write(df, name)


class FolderLibrary(Library):
    def __init__(self, path, format = 'parquet', partitions = 100):
        super(FolderLibrary, self).__init__()
        self.path = path
        self.format = format
        self.partitions = partitions

    def _get_name(self, name):
        if not isinstance(name, str): raise TypeError('Invalid type. Expected `str` but got `{type}`.'.format(type=type(name)))
        elif not re.match(r'^[a-zA-Z0-9_.]*$', name): raise ValueError('Invalid value: {name}'.format(name=name))
        else: return name.lower()

    def _get_path(self, name):
        return '{path}/{name}'.format(path = self.path, name=self._get_name(name))

    def read(self, name, **kwargs):
        return spark.read.load(self._get_path(name), format = self.format, **kwargs)

    def write(self, df, name, **kwargs):
        df.repartition(self.partitions).write.save(self._get_path(name), format = self.format, **kwargs)
        return self.read(name)



class DataBaseLibrary(FolderLibrary):

    def __init__(self, spark, path, db, format = 'parquet', partitions = 100):
        super(DataBaseLibrary, self).__init__()
        self.spark = spark
        self.path = path
        self.db = db
        self.format = format
        self.partitions = partitions

    def _get_table_name(self, name):
        return '{db}.{name}'.format(db=self.db, name=self._get_name(name))

    def write(self, df, name, mode = 'error', **kwargs):
        df.reparition(self.partitions).write.saveAsTable(self._get_table_name(name), format=self.format, mode=mode, path=self._get_path(name))
        return self.read(name)

