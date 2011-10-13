"""
Sample problems on which to test algorithms.

"""
import numpy
import theano
from theano import tensor

import base
from hyperopt.ht_dist2 import rSON2, rlist2, one_of, uniform, normal, lognormal, ceil_lognormal
from hyperopt.base import Bandit

def model_from_template(template):
    pass

def get_performance(model,task):
    pass

class EvaluationBandit(Bandit):
    
    @classmethod
    def evaluate(cls,argd,ctrl):
        #stuff goes here to do model evaluation and interface with relevant dbs
        model = model_from_template(argd['model'])
        performance = get_performance(model,argd['task'])
        return dict(performance = performance, status = 'ok')

