from starflow.utils import creates, activate, is_string_like

from collections import OrderedDict
import itertools
import gridfs
import hashlib
import os
import datetime
import time
import random
import cPickle
from starflow.utils import activate
import pymongo as pm



#############general DB things

CERTIFICATE_ROOT = '../.db_certificates'

def initialize_certificates(creates = CERTIFICATE_ROOT):
    MakeDir(creates)


#####high-level functions

def DBAdd(initialize,args = ()):
    """
        main DB protocol generator.   this (and the decorators) are the main
        thing used by the user
    """
    oplist = initialize(*args)   
    db_ops_initialize(oplist)
    D = [(a[0],db_update,(a[1],initialize,args)) for a in oplist]       
    return D
    

def inject(dbname,outroots,generator,setup=None,cleanup=None,caches=None):
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.meta_action = inject_op
        f.action_name = 'inject'
        f.dbname = dbname
        f.inroots = ['']
        f.outroots = outroots
        f.generator = generator
        f.setup = setup
        f.cleanup = cleanup

        return f
        
    return func
    
    
def dot(dbname,inroots,outroots,setup=None,cleanup=None):
    if is_string_like(inroots):
        inroots = [inroots]
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.dbname = dbname
        f.meta_action = dot_op
        f.action_name = 'dot'
        f.inroots = inroots
        f.outroots = outroots
        f.setup = setup
        f.cleanup = cleanup

        return f
    return func
    
    
def cross(dbname,inroots,outroots,setup=None,cleanup=None):
    if is_string_like(inroots):
        inroots = [inroots]
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.dbname = dbname 
        f.meta_action = cross_op
        f.action_name = 'cross'
        f.inroots = inroots
        f.outroots = outroots
        f.setup = setup
        f.cleanup = cleanup
        return f
        
    return func
    
    
    
########main db operations
   
def op_depends(x):
    """
        generates paths of read certificates
    """
    f = x[0]
    config_generators = f.in_config_generators
    configs_list = [config_generator() for config_generator in config_generators]  
    deps = [get_cert_path(f.dbname, r, get_config_string(configs)) for (r,configs) in zip(f.inroots,configs_list)]
    return tuple(deps)    


def op_creates(x):
    """
        generates paths of write certificates
    """
    f = x[0]  
    config_generator = f.out_config_generator
    configs = config_generator()
    config_string = get_config_string(configs)      
    creates = [get_cert_path(f.dbname, root, config_string) for root in f.outroots]
    return tuple(creates)


@activate(lambda x : (), op_creates)
def inject_op(func):
    """
       use "func" to inject new data into a source data collection
    """
    configs = func.out_config_generator()
        
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection()
    db = conn[dbname]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    #ensure_indexes(db,outroots)
    
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}    
    
    config_time = FuncTime(func)

    for config in configs:
        assert isinstance(config,OrderedDict)
        if not already_exists(config,out_fs,config_time):
            results = do_rec(None,  config, func, pass_args)
    
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,config)
                fs.put(res,**outdata)
                
    if func.cleanup:
        func.cleanup()   
        
    write_outcerts(func,configs)

        
@activate(op_depends,op_creates)
def dot_op(func):
    """
        takes "zip" of source collection parameters in computing output collections
    """

    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection()
    db = conn[dbname]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    #ensure_indexes(db,outroots)
    ftime = FuncTime(func)
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}

    params = func.params
    if params is None:
        params = OrderedDict([])
        
    config_generators = func.in_config_generators    
    config_list = [config_generator() for config_generator in config_generators]

    check_incerts(func,config_list)
    config_list = zip(*config_list)
    newconfigs = []

    for config_tuple in config_list:         
        filenames = [get_filename(config) for config in config_tuple]
        fhs = [fs.get_version(filename) for (fs,filename) in zip(in_fs,filenames)]
        config_time = max(ftime,*[get_time(fh.upload_date) for fh in fhs])
        
        newconfig = dict_union(config_tuple)           
        newconfig.update(params)
        
        newconfigs.append(newconfig)
        if not already_exists(newconfig,out_fs,config_time):
            results =  do_rec(fhs, newconfig, func, pass_args) 
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,newconfig)
                print("HERE")
                fs.put(res,**outdata)
                print("HERE2") 
    if func.cleanup:
        func.cleanup()     
        
    write_outcerts(func,newconfigs)
    
    
@activate(op_depends,op_creates)
def cross_op(func):
    """
        takes "product" of source collection parameters in computing output collections
    """
    
    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection()
    db = conn[dbname]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]    
    #ensure_indexes(db,outroots)
    ftime = FuncTime(func)      
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}
  
    config_generators = func.in_config_generators
    configs_list = [config_generator() for config_generator in config_generators]
    
    check_incerts(func,configs_list)
        
    config_product = itertools.product(*configs_list)
    params = func.params
    if params is None:
        params = OrderedDict([])    
    
    newconfigs = []
    for config_tuple in config_product:

        filenames = [get_filename(config) for config in config_tuple]
        fhs = [fs.get_version(filename) for (fs,filename) in zip(in_fs,filenames)]
        config_time = max(ftime,*[get_time(fh.upload_date) for fh in fhs])
        
        config_tuple = tuple(list(config_tuple) + [params])
        flat_config = dict_union(config_tuple)
        
        newconfigs.append(flat_config)
        if not already_exists(flat_config,out_fs,config_time):
            results = do_rec(fhs, config_tuple, func, pass_args)
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,flat_config)
                fs.put(res,**outdata)
     

    if func.cleanup:
        func.cleanup()  
        
    write_outcerts(func,newconfigs)


#######technical dependencies
from starflow.utils import get_argd, is_string_like

def get_dep(x,att):
    deps = []
    if hasattr(x[0].meta_action,att):
        deps += getattr(x[0].meta_action,att)(x)
    
    if hasattr(x[1],att):
        args = get_argd(x[2]) 
        deps += getattr(x[1],att)(args)
    return tuple(deps)

@activate(lambda x : get_dep(x,'__dependor__'), lambda x : get_dep(x,'__creator__'))
def db_update(func,initialize,args):
    oplist = initialize(*args)
    db_ops_initialize(oplist)
    meta_action = func.meta_action
    meta_action(func)
    
def get_op_gen(op,oplist):
    func = op[1]
    if not hasattr(func,'out_config_generator'):
        inroots = func.inroots
        if func.action_name == 'inject':
            args = op[2]['args']     
            func.out_config_generator = lambda : func.generator(*args)
            
        elif func.action_name == 'dot':
            if len(op) > 2 and 'params' in op[2]:
                params = op[2]['params']
            else:
                params = OrderedDict([])
            
            func.params = params
            
            parents = []
            for ir in inroots:
                try:
                    parent = [op0 for op0 in oplist if ir in op0[1].outroots][0]
                except IndexError:
                    raise IndexError, 'No parent found for at least one collection in ' + repr(op0[1].outroots) 
                else:
                    parents.append(parent)
  
            for parent in parents:
                get_op_gen(parent,oplist)
                
            parent_func = parent[1]
            
            func.in_config_generators = [parent[1].out_config_generator for parent in parents]
            
            def newf():
                C = map(dict_union,zip(*[f() for f in func.in_config_generators]))
                for c in C:
                    c.update(params)
                return C
            
            func.out_config_generator = newf
                  
        elif func.action_name == 'cross':
            if len(op) > 2 and 'params' in op[2]:
                params = op[2]['params']
            else:
                params = OrderedDict([])
            
            func.params = params        
            
            parents = []
            for ir in inroots:
                try:
                    parent = [op0 for op0 in oplist if ir in op0[1].outroots][0]
                except IndexError:
                    raise IndexError, 'No parent found for at least one collection in ' + repr(op0[1].outroots)
                else:
                    parents.append(parent)
            
            for parent in parents:
                get_op_gen(parent,oplist)
                
            func.in_config_generators = [parent[1].out_config_generator for parent in parents]
                
            def newf():
                C =  map(dict_union,itertools.product(*[f() for f in func.in_config_generators]))
                for c in C:
                    c.update(params)
                return C
                
            func.out_config_generator = newf
            

#######utils

def db_ops_initialize(oplist):
    for op in oplist:
        print('Initializing', op)
        get_op_gen(op,oplist)
        

def check_incerts(func,configs_list):

    config_strings = [get_config_string(configs) for configs in configs_list]
    
    incertpaths = [get_cert_path(func.dbname, root, s) for (root,s) in zip(func.inroots,config_strings)]
    incertdicts =  [cPickle.load(open(incertpath)) for incertpath in incertpaths]
    assert all([d['db'] == func.dbname and d['root'] == coll and d['configs'] == s for (coll,s,d) in zip(func.inroots,config_strings,incertdicts)])
   
   
def write_outcerts(func,configs):
    config_string = get_config_string(configs)
    outcertpaths = [get_cert_path(func.dbname, root, config_string) for root in func.outroots]
    for (outcertpath,outroot) in zip(outcertpaths,func.outroots):
        createCertificateDict(outcertpath,{'db':func.dbname, 'root':outroot, 'configs':config_string})    
        

def createCertificateDict(path,d,tol=10000000000):
    d['__certificate__'] = random.randint(0,tol)
    dir = os.path.split(path)[0]
    os.makedirs2(dir)
    F = open(path,'w')
    cPickle.dump(d,F)
    F.close()
    
  
def do_rec(in_fhs,config,func,pass_args):
    print("Computing", config)
    if in_fhs:
        results = func(in_fhs,config,**pass_args)          
    else:
        results =  func(config,**pass_args)
    
    if is_string_like(results):
        results = [results]
        
    return results
    

def dict_union(dictlist):
    newdict = dictlist[0].copy()
    for d in dictlist[1:]:
        newdict.update(d)
        
    return newdict 
    
    
def get_time(dt):
    return time.mktime(dt.timetuple()) + dt.microsecond*(10**-6) 
    
from starflow import linkmanagement, storage
def FuncTime(func):
    modulename = func.__module__
    funcname = func.__name__
    modulepath = '../' + modulename + '.py'
    fullfuncname = modulename + '.' + funcname
    Seed = [fullfuncname]
    Up = linkmanagement.UpstreamLinks(Seed)
    check = zip(Up['SourceFile'],Up['LinkSource']) + [(modulepath,funcname)]
    checkpaths = Up['SourceFile'].tolist() + [modulepath]
    times = storage.ListFindMtimes(check,depends_on = tuple(checkpaths))
    return time.mktime(time.gmtime(max(times.values())))


def reach_in(attr,q):
    q1 = OrderedDict([])
    for k in q:
        q1[attr + '.' + k] = q[k]
    return q1   
    
def ensure_indexes(db,roots):
    colls = [db[coll + '.files'] for coll in roots]
    for coll in colls:
        coll.ensure_index('config',unique=True)


def interpret_res(res,config):
    outdata = {'config' : config.copy()}

    outdata['filename'] = get_filename(config) 
    if not isinstance(res,str):
        assert isinstance(res,dict) and 'summary' in res.keys() and 'res' in res.keys()
        summary = res['summary']
        res = res['res']
        outdata['summary'] = summary
                
    return outdata,res
    

def already_exists(config,fs_list,t):

    q = reach_in('config',config)
    d = datetime.datetime.fromtimestamp(t)
    q['uploadDate'] = {'$gte':d}
    exists = [fs.exists(q) for fs in fs_list]
    assert (not any(exists)) or all(exists)
    return all(exists)


def get_cert_path(dbname,root,config_string):
    return os.path.join(CERTIFICATE_ROOT,dbname,root,config_string)
    
def get_config_string(configs):
    return hashlib.sha1(repr(list(configs))).hexdigest()
    
def get_filename(config):
    return hashlib.sha1(repr(config)).hexdigest()    
