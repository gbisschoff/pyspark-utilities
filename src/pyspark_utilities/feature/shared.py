from pyspark.ml.param import *


class HasOrder(Params):

    order = Param(Params._dummy(), "order", "order", typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasOrder, self).__init__()

    def getOrder(self):
        return  self.getOrDefault(self.order)

    def setOrder(self, value):
        return self._set(order=value)


class HasKnots(Params):

    knots = Param(Params._dummy(), "knots", "knots", typeConverter=TypeConverters.toVecor)

    def __init__(self):
        super(HasKnots, self).__init__()

    def getKnots(self):
        return self.getOrDefault(self.knots)

    def setKnots(self, value):
        return self._set(knots=list(value))


class HasIntercept(Params):

    intercept = Param(Params._dummy(), "intercept", "intercept", typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasIntercept, self).__init__()

    def getIntercept(self):
        return self.getOrDefault(self.intercept)

    def setIntercept(self):
        return self._set(intercept=value)

