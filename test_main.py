#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import cPickle

import boto
import gridfs
import scipy as sp

from starflow.utils import MakeDir
from starflow.protocols import protocolize, actualize

from utils import createCertificateDict, wget
from dbutils import connect_to_db 
from v1like_extract import feature_extraction_protocol


@protocolize()
def test_v1like_feature_instantiator():
    feature_extraction_protocol('../config_model.py')


from svm import classify, ova_classify, multi_classify
from splits import generate_split, generate_multi_split

def test_v1like_a_results_on_human_faces(creates = '../human_faces_results/'):
    train_test({'type':'face','subject':'human','mode':'photo'},20,60,N=15,universe={'$or':[{'type':'face','subject':'human','mode':'photo'},{'type':'object'}]})
   

def train_test(outdir,query,ntrain,ntest,ntrain_pos = None,classifier = None,classifier_kwargs = {},N=10,universe=None):
    MakeDir(outdir)

    if isinstance(query,dict):
        splitter = generate_split
        classifier = classify
    else:
        splitter = generate_multi_split
        if classifier is None:
            classifier = ova_classify
               
    split_data = []
    results = []

    for i in range(N):
        print(i)
        split = splitter('v1','extracted_features',query,ntrain,ntest,ntrain_pos = ntrain_pos, universe=universe)
        train_data = split['train_data']
        train_features = split['train_features']
        train_labels = split['train_labels']
        test_data = split['test_data']
        test_features = split['test_features']
        test_labels = split['test_labels']

        if (not classifier_kwargs.get('multi_class')) or len(uniqify(train_labels)) > 2:
            train_filenames = [t['filename'] for t in train_data]
            test_filenames = [t['filename'] for t in test_data]
            split_data.append({'train_filenames':train_filenames,'train_labels':train_labels,
                           'test_filenames': test_filenames,'test_labels':test_labels})
                           
            assert set(train_filenames).intersection(test_filenames) == set([])
            res = classifier(train_features,train_labels,test_features,test_labels,**classifier_kwargs)
        
            results.append(res)

    stats = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']
    
    output = {'split_results' : results}
    
    for stat in stats:
        if stat in results[0] and results[0][stat] != None:
            output[stat] = sp.array([result[stat] for result in results]).mean()
    

    F = open(os.path.join(outdir,'splits.pickle'),'w')
    cPickle.dump(split_data,F)
    F.close()
    F = open(os.path.join(outdir,'results.pickle'),'w')
    cPickle.dump(output,F)
    F.close()
    


    