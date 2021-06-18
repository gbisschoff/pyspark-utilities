import time
import datetime


"""
Timer function meant to be used as a decorator
"""
def timer(func):
    def wrapper(*args,**kwargs):
        before = time.time()
        val = func(*args,**kwargs)
        total_time = str(datetime.timedelta(seconds=time.time()-before))
        list_time = total_time.split(':')
        print("Function {func_name} took:".format(func_name = func.__name__), list_time[-3], "hours,", list_time[-2], "minutes and", list_time[-1], "seconds to run.")
        return val
    return wrapper

