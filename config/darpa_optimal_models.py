#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy
import pymongo as pm
import numpy as np
from bson import SON

model_filename = '3246eb8a8012508635a12c9b48be7fb1f5fbafca'

conn = pm.Connection(document_class=SON)
db = conn['thor']
coll = db['models.files']

model = coll.find_one({'filename':model_filename})['config']['model']

config = {
     'models': [model]
}
 





