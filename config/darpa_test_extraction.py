#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

import copy

from bson import SON


extraction = SON([('transform_average',SON([('transform_name','translation'),('percentile',[0,10,20,80,90,100])])),
             ])

config = {
     'extractions': [extraction]
}
 





