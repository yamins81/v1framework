#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy
import pymongo as pm
import numpy as np
from bson import SON

hash = '14779eea1f68449231f21ff3b5c7ceaab82e7369'
conn = pm.Connection(document_class=SON)
db = conn['thor']
coll = db['performance']
L = list(coll.find({'__hash__':hash},fields=['test_accuracy','model']))
p = np.array( [l['test_accuracy'] for l in L])
L1 = [L[i] for i in p.argsort()]
Top5 = [l['model'] for l in L1[-5:]]

config = {
     'models': Top5
}
 





