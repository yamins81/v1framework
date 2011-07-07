import pymongo as pm
conn = pm.Connection()
coll = conn['thor']['performance']
import numpy as np
import scipy.stats as stats

def get_perfq(hash,q,p=75):
    q['__hash__'] = hash
    res = np.array([l['test_accuracy'] for l in coll.find(q,fields=['test_accuracy'])])
    return lof([res.max(),res.min(),res.mean(),stats.scoreatpercentile(res,p),res.std()])

def get_perf(hash,level,key,att):
    level = str(level)
    q = {'__hash__':hash}
    vals = coll.find(q).distinct('model.layers.' + level + '.' + key + '.' + att)

    maxs = []
    mins = []
    means = []
    quartiles = []
    stds = []
    for val in vals:
        q['model.layers.' + level + '.' + key + '.' + att] = val
        res = np.array([l['test_accuracy'] for l in coll.find(q,fields=['test_accuracy'])])
        maxs.append(res.max())
        means.append(res.mean())
        mins.append(res.min())
        quartiles(stats.scoreatepercentile(res,.75))
        stds.append(res.std())

    print('level %s, %s, %s' % (level,key,att))
    print('value: %s' % repr(vals))
    print('max: %s' % lof(maxs))
    print('mean: %s' % lof(means))
    print('quartiles %s' % lof(quartiles))
        
    return maxs,mins,means,quartiles,stds

def lof(mylist):
    return ", ".join(['%0.2f' % item for item in mylist])
