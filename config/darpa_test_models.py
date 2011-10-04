#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy

from bson import SON



model = SON([('selection','random'),
             ('generation_function_path','darpa.generate_random_model'),
             ('num_models',100)])

config = {
     'models': [model]
}
 





