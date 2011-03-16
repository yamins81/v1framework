from starflow.utils import creates, activate, is_string_like, ListUnion, uniqify, get_argd, is_string_like

#from collections import OrderedDict
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
from bson import SON,BSON



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
    

def aggregate(dbname,inroots,aggregate_on,outroots,setup=None,cleanup=None):   
    if is_string_like(inroots):
        inroots = [inroots]
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.dbname = dbname 
        f.aggregate_on = aggregate_on
        f.meta_action = aggregate_op
        f.action_name = 'aggregate'
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
    deps = [get_cert_path(f.dbname, r, get_config_string(args)) for (r,args) in zip(f.inroots,f.in_args)]
    return tuple(deps)    


def op_creates(x):
    """
        generates paths of write certificates
    """
    f = x[0]  
    config_string = get_config_string(f.out_args) 
    creates = [get_cert_path(f.dbname, root, config_string) for root  in f.outroots]
    return tuple(creates)


@activate(lambda x : (), op_creates)
def inject_op(func):
    """
       use "func" to inject new data into a source data collection
    """
    configs = func.config_generator()
        
    config_str = get_config_string(func.out_args)    
        
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection(document_class=SON)
    db = conn[dbname]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    out_cols = [db[coll + '.files'] for coll in outroots]
    
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}    
    
    config_time = FuncTime(func)

    for config in configs:
        assert isinstance(config,SON)
        if not already_exists(config,out_cols,config_time,config_str):
            results = do_rec(None,  config, func, pass_args)
    
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,config,config_str)
                fs.put(res,**outdata)
                
    if func.cleanup:
        func.cleanup()   
        
    write_outcerts(func,configs,None)

        
@activate(op_depends,op_creates)
def dot_op(func):
    """
        takes "zip" of source collection parameters in computing output collections
    """

    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection(document_class=SON)
    db = conn[dbname]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    in_cols = [db[coll + '.files'] for coll in inroots]
    out_cols = [db[coll + '.files'] for coll in outroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    
    ftime = FuncTime(func)
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}

    params = func.params
    if params is None:
        params = SON([])
    
    inconfig_strings = [get_config_string(a) for a in func.in_args]
    outconfig_string = get_config_string(func.out_args)
    
    incertdicts = check_incerts(func)    
        
    data_list = get_data_list(in_cols,inconfig_strings)
    data_list = zip(*data_list)

    newdata_list = []
    for data_tuple in data_list:         
        filenames = [dt['filename'] for dt in data_tuple]
        data_tuple = [dt['config'] for dt in data_tuple]
        fhs = [fs.get_version(filename) for (fs,filename) in zip(in_fs,filenames)]
        config_time = max(ftime,*[get_time(fh.upload_date) for fh in fhs])
        
        newdata = dict_union(data_tuple)
        newdata.update(params)
        newdata_list.append(newdata)
 
        if not already_exists(newdata,out_cols,config_time,outconfig_string):
            results =  do_rec(fhs, newdata, func, pass_args) 
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,newdata,outconfig_string)
                fs.put(res,**outdata)
                 
    if func.cleanup:
        func.cleanup()     
        
    write_outcerts(func,newdata_list,incertdicts)
    

    
@activate(op_depends,op_creates)
def cross_op(func):
    """
        takes "product" of source collection parameters in computing output collections
    """
    
    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection(document_class=SON)
    db = conn[dbname]
    in_cols = [db[coll + '.files'] for coll in inroots]
    out_cols = [db[coll + '.files'] for coll in outroots]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]    
    ftime = FuncTime(func)      
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}


    inconfig_strings = [get_config_string(a) for a in func.in_args]
    outconfig_string = get_config_string(func.out_args)
    incertdicts = check_incerts(func)    
    
    data_list = get_data_list(in_cols,inconfig_strings)
    params = func.params
    if params is None:
        params = SON([])
            
    data_product = itertools.product(*data_list)
    
    newdata_list = []
    for data_tuple in data_product:
        filenames = [dt['filename'] for dt in data_tuple]
        data_tuple = [dt['config'] for dt in data_tuple]
        fhs = [fs.get_version(filename) for (fs,filename) in zip(in_fs,filenames)]
        config_time = max(ftime,*[get_time(fh.upload_date) for fh in fhs])
        
        data_tuple = tuple(list(data_tuple) + [params])
        flat_data = dict_union(data_tuple)        
        newdata_list.append(flat_data)

        if not already_exists(flat_data,out_cols,config_time,outconfig_string):
            results = do_rec(fhs, data_tuple, func, pass_args)
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,flat_data,outconfig_string)
                fs.put(res,**outdata)
     
    if func.cleanup:
        func.cleanup()  
        
    write_outcerts(func,newdata_list,incertdicts)


    
@activate(op_depends,op_creates)
def aggregate_op(func):

    aggregate_on = func.aggregate_on
    
    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = pm.Connection(document_class=SON)
    db = conn[dbname]
    in_cols = [db[coll + '.files'] for coll in inroots]
    out_cols = [db[coll + '.files'] for coll in outroots]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]    

    ftime = FuncTime(func)      
    if func.setup:
        pass_args = func.setup() or {}
    else:
        pass_args = {}
  
 
    inconfig_strings = [get_config_string(a) for a in func.in_args]
    outconfig_string = get_config_string(func.out_args)
    incertdicts = check_incerts(func)    
    
    data_list = get_data_list(in_cols,inconfig_strings)
    params = func.params
    if params is None:
        params = SON([])   

    aggregate_params = old_params[aggregate_on]
    aggregate_val = func.args[aggregate_on]

  
    D = get_aggregate(data_list,aggregate_val,aggregate_params,aggregate_on,params)
         
    for (dtuples,Ndict) in D.values():
        
        filenames = [[data['filename'] for data in data_tuple] for data_tuple in dtuples]
        filehandles = [[fs.get_version(filename) for (fs,fname) in zip(in_fs,fnames)] for fnames in filenames]
        config_time = max([max(ftime,*[get_time(fh.upload_date) for fh in fhs]) for fhs in filehandles])
               
        if not already_exists(newconfig,out_cols,config_time,outconfig_str):
            results =  do_rec(filehandles, dtuples, func, pass_args) 
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,Ndict,outconfig_string)
                fs.put(res,**outdata)
         
    if func.cleanup:
        func.cleanup()      
        
    newdata_list = [x[0] for x in D.values()]    
    write_outcerts(func,newdata_list,incertdicts)


def get_aggregate(config_list,aggregate_val,aggregate_params,aggregate_on,params):
    config_list = zip(*config_list)
    D = {}
    for config_tuple in config_list:
        N = []
        config_tuple = [c['config'] for c in config_tuple]
        for c in config_tuple:
            nonagg_params = set(c.keys()).difference(aggregate_params)
            nonagg_values = SON([(p,c[p]) for p in nonagg_params])
            N.append(nonagg_values)
        Ndict = dict_union(N)
        Ndict[aggregate_on + '__aggregate__'] = aggregate_val 
        Ndict.update(params)
        
        r = repr(NDict)
        if r in D:
            D[r].append((config_tuple,Ndict))
        else:
            D[r] = [(config_tuple,Ndict)]
            
    return D


#######technical dependencies

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
    if not hasattr(func,'out_args'):
        inroots = func.inroots
        outroots = func.outroots
        if func.action_name == 'inject':
            args = op[2]['args']  
            #check_args(args)
            
            func.config_generator = lambda : func.generator(*args)
            func.out_args = SON([(outroot,args) for outroot in outroots])
            print('OA',func.out_args)
            
        else:
            if len(op) > 2 and 'params' in op[2]:
                params = op[2]['params']
            else:
                params = SON([])
            
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
                
            func.in_args = [parent[1].out_args for parent in parents]
            outargs = dict_union(func.in_args)
            outargs.update(params)
            func.out_args = outargs

            

#######utils
def check_args(args):
    BSON.encode(args,check_keys=True)

def db_ops_initialize(oplist):
    for op in oplist:
        print('Initializing', op)
        get_op_gen(op,oplist)
        

def check_incerts(func):    
    config_strings = [get_config_string(a) for a in func.in_args]
    incertpaths = [get_cert_path(func.dbname, root, s) for (root,s) in zip(func.inroots,config_strings)]
    incertdicts =  [cPickle.load(open(incertpath)) for incertpath in incertpaths]
    assert all([d['db'] == func.dbname and d['root'] == coll and d['configs'] == s for (coll,s,d) in zip(func.inroots,config_strings,incertdicts)])
    return incertdicts

   
def write_outcerts(func,configs,incertdicts):
    if incertdicts:
        old_param_names = dict_union([op['param_names'] for op in incertdicts])
    else:
        old_param_names = SON([])
    config_string = get_config_string(func.out_args)
    new_param_names = uniqify(ListUnion([x.keys() for x in configs]))
    
    outcertpaths = [get_cert_path(func.dbname, root, config_string) for root in func.outroots]
    for (outcertpath,outroot) in zip(outcertpaths,func.outroots):
        param_names = old_param_names.copy()
        param_names[outroot] = new_param_names
        createCertificateDict(outcertpath,{'db':func.dbname, 'root':outroot, 'configs':config_string, 'param_names':param_names})    
        

def createCertificateDict(path,d,tol=10000000000):
    d['__certificate__'] = random.randint(0,tol)
    dir = os.path.split(path)[0]
    os.makedirs2(dir)
    F = open(path,'w')
    cPickle.dump(d,F)
    F.close()
    
  
def do_rec(in_fhs,config,func,pass_args):
    print(config)
    if in_fhs:
        results = func(in_fhs,config,**pass_args)          
    else:
        results =  func(config,**pass_args)
    
    if not (isinstance(results,tuple) or isinstance(results,list)):
        results = [results]
        
    return results
    

def dict_union(dictlist):
    newdict = dictlist[0].copy()
    for d in dictlist[1:]:
        newdict.update(d)
        
    return newdict 
    
def get_time(dt):
    #return time.mktime(dt.timetuple()) + dt.microsecond*(10**-6) 
    return dt.timetuple()
    
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
    return time.gmtime(max(times.values()))

def reach_in(attr,q):
    q1 = SON([])
    for k in q:
        q1[attr + '.' + k] = q[k]
    return q1   
    

def interpret_res(res,data,cstr):
    
    datacopy = data.copy()
    
    if not isinstance(res,str):
        print("HERE")
        assert isinstance(res,dict)
        file_res = res.pop('__file__','')
        datacopy.update(res)
    else:
        file_res = res
        
    outdata = {'config' : datacopy, '__hash__' : [cstr]}
    outdata['filename'] = get_filename(data) 

    return outdata,file_res
    
    
def already_exists(config,coll_list,t, cstr):
    q = reach_in('config',config)
    #d = datetime.datetime.fromtimestamp(t)
    d = datetime.datetime(*t[:6])
    q['uploadDate'] = {'$gte':d}
    recs = [coll.find_one(q) for coll in coll_list]
    
    all_exist = all(recs)
    
    assert (not any(recs)) or all_exist
    
    if all_exist:
        for (coll,rec) in zip(coll_list,recs):
            if cstr not in rec['__hash__']:
                coll.update(q,{'$addToSet':{'__hash__':cstr}})   #the relevant mongo thing
    
    return all_exist

def get_cert_path(dbname,root,config_string):
    return os.path.join(CERTIFICATE_ROOT,dbname,root,config_string)
    
def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()
    
def get_filename(config):
    return hashlib.sha1(repr(config)).hexdigest()    

def get_data_list(in_cols,inconfig_strings):
    return [get_most_recent_files(in_col,{'__hash__':cstr},kwargs={'fields':['config','filename']}) for (in_col,cstr) in zip(in_cols,inconfig_strings)]

def get_most_recent_files(coll,q,kwargs=None):
    if kwargs is None:
        kwargs = {}
    c = coll.find(q,**kwargs).sort([("filename", 1), ("uploadDate", -1)])    
    cl = list(c)
    return get_recent(cl)
    
    
def get_recent(filespecs):
    return [f for (i,f) in enumerate(filespecs) if i == 0 or filespecs[i-1]['filename'] != f['filename']]