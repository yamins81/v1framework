import urllib
import json
import cPickle
import Image
import os

from dbutils import get_config_string as gcs

def get_files():
    url = 'http://50.19.109.25:9999/db/thor/correlation_extraction.files?query={"__hash__":"768bd704554d4758f578ca6b83413fd9bae736e8"}&fields=["filename","model"]'
    
    X = json.loads(urllib.urlopen(url).read())
    url2 = 'http://50.19.109.25:9999/db/thor/performance?query={"__hash__":"0681f20ebdbb3c6b4feff9f17840eb5b41e739a3"}&fields=["model","test_accuracy"]'
    X2 = json.loads(urllib.urlopen(url2).read())
    for x in X2:
    	x['model']['layers'].pop(2)
    mhashes = dict([(gcs(x['model']),x['test_accuracy']) for x in X2])

    for x in X:
        print x['filename']
        mh = gcs(x['model'])
        ta = mhashes[mh]
        url = 'http://50.19.109.25:9999/file/thor/correlation_extraction/' + x['filename']
        y = urllib.urlopen(url).read()
        F = open(os.path.join('testdir',x['filename'] + '.' + str(ta) +  '.pickle'),'w')
        F.write(y)
        F.close()
        V,M = cPickle.loads(y)['sample_result']
        I = Image.fromarray(1000*V)
        I.save(os.path.join('testdir',x['filename'] + '.' + str(ta) + '.tif'))
        
import pymongo as pm
import gridfs

def get_files2():
    
    url = 'http://50.19.109.25:9999/db/thor/correlation_extraction.files?query={"__hash__":"768bd704554d4758f578ca6b83413fd9bae736e8"}&fields=["filename","model"]'    
    X = json.loads(urllib.urlopen(url).read())
    url2 = 'http://50.19.109.25:9999/db/thor/performance?query={"__hash__":"0681f20ebdbb3c6b4feff9f17840eb5b41e739a3"}&fields=["model","test_accuracy"]'
    X2 = json.loads(urllib.urlopen(url2).read())
    for x in X2:
    	x['model']['layers'].pop(2)
    mhashes = dict([(gcs(x['model']),x['test_accuracy']) for x in X2])

    conn = pm.Connection()
    db = conn['thor']
    fs = gridfs.GridFS(db,'correlation_extraction')
    
    for x in X:
        print x['filename']
        mh = gcs(x['model'])
        ta = mhashes[mh]
        y = fs.get_version(x['filename']).read()
        V,M = cPickle.loads(y)['sample_result']
        I = Image.fromarray(1000*V)
        I.save(os.path.join('testdir',str(ta) + '.' + x['filename'] + '.tif'))
        
        