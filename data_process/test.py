# -*- coding:utf-8 -*-


from multiprocessing import Pool
import numpy as np

def func(a, b):
    return a+b

p = Pool()
a = [1,2,3]
b = [4,5,6]
para = zip(a,b)
result = p.map(func, para)
p.close()
p.join()
print result