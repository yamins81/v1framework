import cPickle
import pymongo as pm
import scipy as sp
from bson import SON
import gridfs
from starflow.utils import uniqify, ListUnion

from svm import classify, ova_classify, multi_classify


"""
train / test 
"""


def train_test(query,dbname, colname, ntrain,ntest,ntrain_pos = None,classifier = None,classifier_kwargs = None,N=10,universe=None):

    print('Q',query)
    print('U',universe)

    if classifier_kwargs is None:
        classifier_kwargs = {}

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
        split = splitter(dbname,colname,query,ntrain,ntest,ntrain_pos = ntrain_pos, universe=universe)
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
    
    outdata = SON([('split_results' , results),('split_data',split_data)])
    
    outresults = SON([])
    for stat in stats:
        if stat in results[0] and results[0][stat] != None:
            outresults[stat] = sp.array([result[stat] for result in results]).mean()
    

    return outdata, outresults


def combine_things(a,b):
    for k in b:
        if k == '$where' and k in a:
            a[k] = a[k].strip('; ') + ' && ' + b[k]
                
        elif k == '$or' and k in a:
            pass
        elif hasattr(b[k],'get') and (b[k].get('$in') or b[k].get('$nin') or b[k].get('$ne')):
            pass
            
        else:
            a[k] = b[k]
    

from dbutils import get_most_recent_files, dict_union
def generate_split(dbname,collectionname,task_query,ntrain,ntest,ntrain_pos = None, universe = None):

    if universe is None:
        universe = SON([])

    connection = pm.Connection(document_class=SON)
    db = connection[dbname]
    data = db[collectionname + '.files']
    fs = gridfs.GridFS(db,collection=collectionname)

    combine_things(task_query,universe)
    
    task_data = get_most_recent_files(data,task_query)
    task_fnames = [x['filename'] for x in task_data]
    N_task = len(task_data)
    
    nontask_query = {'filename':{'$nin':task_fnames}}
    nontask_query.update(universe)
    nontask_data = get_most_recent_files(data,nontask_query)
    N_nontask = len(nontask_data)

    assert ntrain + ntest <= N_task + N_nontask, "Not enough training and/or testing examples " + str([N_task,N_nontask])
      
    if ntrain_pos is not None:
        ntrain_neg = ntrain - ntrain_pos
        assert ntrain_pos <= N_task, "Not enough positive training examples, there are: " + str(N_task)
        assert ntrain_neg <= N_nontask, "Not enough negative training examples, there are: " + str(N_nontask)
        
        perm_pos = sp.random.permutation(len(task_data))
        perm_neg = sp.random.permutation(len(nontask_data))
        
        train_data = [task_data[i] for i in perm_pos[:ntrain_pos]] + [nontask_data[i] for i in perm_neg[:ntrain_neg]]
        
        all_test = [task_data[i] for i in perm_pos[ntrain_pos:]] + [nontask_data[i] for i in perm_neg[ntrain_neg:]]
        
        new_perm = sp.random.permutation(len(all_test))
        
        test_data = [all_test[i] for i in new_perm[:ntest]]
        
    
    else:
        
        all_data = task_data + nontask_data
         
        perm = sp.random.permutation(len(all_data))
         
        train_data = [all_data[i] for i in perm[:ntrain]]
    
        test_data = [all_data[i] for i in perm[ntrain:ntrain + ntest]]
        
     
    train_labels = sp.array([x['filename'] in task_fnames for x in train_data])
    test_labels = sp.array([x['filename'] in task_fnames for x in test_data])
    
    train_features = sp.row_stack([cPickle.loads(fs.get_version(r['filename']).read()) for r in train_data])
    test_features = sp.row_stack([cPickle.loads(fs.get_version(r['filename']).read()) for r in test_data])
    
    return {'train_data': train_data, 'test_data' : test_data, 'train_features' : train_features,'train_labels':train_labels,'test_features':test_features,'test_labels':test_labels}

import numpy as np
import tabular as tb    
import copy
def generate_split2(dbname,collectionname,task_query,N,ntrain,ntest,ntrain_pos = None, ntest_pos = None, universe = None,use_negate = False):

    task_query = copy.deepcopy(task_query)
    print('Generating splits ...')
    if universe is None:
        universe = SON([])

    connection = pm.Connection(document_class=SON)
    db = connection[dbname]
    data = db[collectionname + '.files']

    fs = gridfs.GridFS(db,collection=collectionname)

    combine_things(task_query,universe)
    
    print('T',task_query)
    task_data = get_most_recent_files(data,task_query)
    task_fnames = [str(x['filename']) for x in task_data]
    N_task = len(task_data)
    
    if use_negate:
        task_fnames = np.array(task_fnames)
        all_data = get_most_recent_files(data,universe)
        all_fnames = np.array([str(x['filename']) for x in all_data])
        I = np.invert(tb.isin(all_fnames,task_fnames)).nonzero()[0]
        nontask_data = [all_data[ind] for ind in I]
        nontask_fnames = [str(x['filename']) for x in nontask_data]
        assert set(task_fnames).intersection(nontask_fnames) == set([]), set(task_fnames).intersection(nontask_fnames)
    else:
        nontask_query = {'filename':{'$nin':task_fnames}}
        nontask_query.update(universe)
        nontask_data = get_most_recent_files(data,nontask_query)
        
    N_nontask = len(nontask_data)

    assert ntrain + ntest <= N_task + N_nontask, "Not enough training and/or testing examples " + str([N_task,N_nontask])
      
    splits = []  
    for ind in range(N):
        print('... split', ind)
        if ntrain_pos is not None:
            ntrain_neg = ntrain - ntrain_pos
            assert ntrain_pos <= N_task, "Not enough positive training examples, there are: " + str(N_task)
            assert ntrain_neg <= N_nontask, "Not enough negative training examples, there are: " + str(N_nontask)
            
            perm_pos = sp.random.permutation(len(task_data))
            perm_neg = sp.random.permutation(len(nontask_data))
            
            train_data = [task_data[i] for i in perm_pos[:ntrain_pos]] + [nontask_data[i] for i in perm_neg[:ntrain_neg]]    
            
            if ntest_pos is not None:
                ntest_neg = ntest - ntest_pos
                assert ntest_pos <= N_task - ntrain_pos, "Not enough positive test examples, there are: " + str(N_task - ntrain_pos)
                assert ntest_neg <= N_nontask - ntrain_neg, "Not enough negative test examples, there are: " + str(N_nontask - ntrain_neg)       
                test_data = [task_data[i] for i in perm_pos[ntrain_pos:ntrain_pos + ntest_pos]] + [nontask_data[i] for i in perm_neg[ntrain_neg:ntrain_neg + ntest_neg]]          
            else:     
                nontrain_data = [task_data[i] for i in perm_pos[ntrain_pos:]] + [nontask_data[i] for i in perm_neg[ntrain_neg:]]
                new_perm = sp.random.permutation(len(nontrain_data))
                test_data = [nontrain_data[i] for i in new_perm[:ntest]]
            
        
        else:
            if ntest_pos is not None:
                ntest_neg = ntest - ntest_pos
                assert ntest_pos <= N_task, "Not enough positive test examples, there are: " + str(N_task)
                assert ntest_neg <= N_nontask, "Not enough negative test examples, there are: " + str(N_nontask)                   
                perm_pos = sp.random.permutation(len(task_data))
                perm_neg = sp.random.permutation(len(nontask_data))
                test_data = [task_data[i] for i in perm_pos[:ntest_pos]] + [nontask_data[i] for i in perm_neg[:ntest_neg]]   
                nontest_data = [task_data[i] for i in perm_pos[ntest_pos:]] + [nontask_data[i] for i in perm_neg[ntest_neg:]]
                new_perm = sp.random.permutation(len(nontest_data))
                train_data = [nontest_data[i] for i in new_perm[:ntrain]]               
            else:
                all_data = task_data + nontask_data
                perm = sp.random.permutation(len(all_data))
                train_data = [all_data[i] for i in perm[:ntrain]]
                test_data = [all_data[i] for i in perm[ntrain:ntrain + ntest]]
            
        train_filenames = np.array([str(_t['filename']) for _t in train_data])
        test_filenames = np.array([str(_t['filename']) for _t in test_data])
        
        train_labels = tb.isin(train_filenames,task_fnames)
        test_labels = tb.isin(test_filenames,task_fnames)
         
        #train_labels = sp.array([x['filename'] in task_fnames for x in train_data])
        #test_labels = sp.array([x['filename'] in task_fnames for x in test_data])

        assert set(train_filenames).intersection(test_filenames) == set([]), str(set(train_filenames).intersection(test_filenames))
        
        split = {'train_data': train_data, 'test_data' : test_data, 'train_labels':train_labels,'test_labels':test_labels}
        splits.append(split)
   
    return splits


def validate(idseq):
    ids = ListUnion(idseq)
    ids1 = [id[1] for id in ids]
    assert len(uniqify(ids1)) == sum([len(X) for X in idseq]), 'Classes are not disjoint.'
    return ids
    
    
def generate_multi_split(dbname,collectionname,queries,ntrain,ntest,ntrain_pos = None, universe = None):

    if universe is None:
        universe = {}
    
    for q in queries:
        q.update(universe)
        
    connection = pm.Connection()
    db = connection[dbname]
    data = db[collectionname]
    
    task_data_list = [list(data.find(query)) for query in queries]
    task_id_list = [[(i,x['_id']) for x in X] for (i,X) in enumerate(task_data_list)]
    task_data = ListUnion(task_data_list)
    task_ids = validate(task_id_list)
    task_dist, task_ids = zip(*task_ids)
    task_dist = list(task_dist) ; task_ids = list(task_ids)
        
    nontask_query = {'_id':{'$nin':task_ids}}    
    nontask_query.update(universe)
    nontask_data = list(data.find(nontask_query)) 
    nontask_ids = [x['_id'] for x in nontask_data]
        
    all_ids = task_ids + nontask_ids
    all_data = task_data + nontask_data
    all_dist = task_dist + [len(queries)]*len(nontask_ids)
    
    assert ntrain + ntest <= len(all_ids)
    
    perm = sp.random.permutation(len(all_ids))
  
    train_ids = [all_ids[i] for i in perm[:ntrain]]
    test_ids = [all_ids[i] for i in perm[ntrain:ntrain + ntest]]
        
    train_data = [all_data[i] for i in perm[:ntrain]]
    test_data = [all_data[i] for i in perm[ntrain:ntrain+ntest]]
    
    train_labels = sp.array([all_dist[i] for i in perm[:ntrain]])
    test_labels = sp.array([all_dist[i] for i in perm[ntrain:ntrain+ntest]]) 

    train_features = sp.row_stack([cPickle.loads(data.fs.get(r).read()) for r in train_ids])
    test_features = sp.row_stack([cPickle.loads(data.fs.get(r).read()) for r in test_ids])
    
    return {'train_data': train_data,
            'test_data' : test_data, 
            'train_features' : train_features,
            'train_labels':train_labels,
            'test_features':test_features,
            'test_labels':test_labels
           }

 
 
def generate_multi_split2(dbname,collectionname,task_queries,N,ntrain,ntest,universe = None,labels=None):

    nq = len(task_queries)
    if labels is None:
        labels = range(nq)
    
    task_queries = [copy.deepcopy(task_query) for task_query in task_queries]
    print('Generating splits ...')
    if universe is None:
        universe = SON([])

    connection = pm.Connection(document_class=SON)
    db = connection[dbname]
    data = db[collectionname + '.files']

    fs = gridfs.GridFS(db,collection=collectionname)

    for task_query in task_queries:
        combine_things(task_query,universe)
    
    task_data = [get_most_recent_files(data,task_query,kwargs={"fields":["filename"]}) for task_query in task_queries]
    
    ntrain_vec = [ntrain/nq]*(nq - 1) + [ntrain - (ntrain/nq)*(nq-1)]
    ntest_vec = [ntest/nq]*(nq - 1) + [ntest - (ntest/nq)*(nq-1)]
    
    for (tq,td,ntr,nte) in zip(task_queries,task_data,ntrain_vec,ntest_vec):
        assert ntr + nte <= len(td), 'not enough examples to train/test for %s, %d needed, but only have %d' % (repr(tq),ntr+nte,len(td))
    
    splits = []  
    for ind in range(N):
        print('... split', ind)
        
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for (label,td,ntr,nte) in zip(labels,task_data,ntrain_vec,ntest_vec):
            perm = sp.random.permutation(len(td))
            train_data.extend([td[i] for i in perm[:ntr]])
            test_data.extend([td[i] for i in perm[ntr:ntr+nte]])
            train_labels.extend([label]*ntr)
            test_labels.extend([label]*nte)

        train_filenames = np.array([str(_t['filename']) for _t in train_data])
        test_filenames = np.array([str(_t['filename']) for _t in test_data])
        assert set(train_filenames).intersection(test_filenames) == set([]), str(set(train_filenames).intersection(test_filenames))
             
        split = {'train_data': train_data, 'test_data' : test_data, 'train_labels':train_labels,'test_labels':test_labels}
        splits.append(split)
   
    return splits