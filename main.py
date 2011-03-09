#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from starflow.protocols import protocolize

from v1like_extract import v1_feature_extraction_protocol
from traintest import train_test

@protocolize()
def test_extract_cairo(depends_on = '../config/config_model2.py'):
    v1_feature_extraction_protocol(depends_on)



#def test_v1like_a_results_on_human_faces(creates = '../human_faces_results/'):
#    train_test({'type':'face','subject':'human','mode':'photo'},20,60,N=15,universe={'$or':[{'type':'face','subject':'human','mode':'photo'},{'type':'object'}]})
   

    


    