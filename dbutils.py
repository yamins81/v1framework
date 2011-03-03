from starflow.utils import creates, activate, is_string_like
from collections import OrderedDict
import itertools
import gridfs
import hashlib
import os
import datetime
import time



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


@activate(lambda x : None, op_creates)
def inject_op(func):
    """
       use "func" to inject new data into a source data collection
    """
    configs = func.out_config_generator()
        
    outroots = func.outroots
    dbname = func.dbname
    conn = connect_to_db()
    db = conn[dbname]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    ensure_indexes(db,outroots)
    
    if func.setup:
        pass_args = func.setup()
    else:
        pass_args = {}    
    
    config_time = FuncTime(func)

    for config in configs:
        assert isinstance(config,OrderedDict)
        if not already_exists(config,out_fs,config_time):
            res = func(config,**pass_args)
    
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
    conn = connect_to_db()
    db = conn[dbname]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    ensure_indexes(db,outroots)
    ftime = FuncTime(func)
    if func.setup:
        pass_args = func.setup()
    else:
        pass_args = {}

    params = func.params
    if params is None:
        params = OrderedDict([])
        
    config_generators = func.in_config_generators    
    config_list = zip(*[config_generator() for config_generator in config_generators])
    newconfigs = []

    check_incerts(f,config_list)

    for config_tuple in config_list:         
        filenames = [get_filename(config) for config in config_tuple]
        fhs = [fs.get_version(filename) for (fs,filename) in zip(in_fs,filenames)]
        config_time = max(ftime,*[get_time(fh.update_date) for fh in fhs])
        
        newconfig = dict_union(config_tuple)           
        newconfig.update(params)
        
        newconfigs.append(newconfig)
        if not already_exists(newconfig,out_fs,config_time):
            results =  func(in_fhs,newconfig,**pass_args)           
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,newconfig)
                fs.put(res,**outdata)
                 
    if func.cleanup:
        func.cleanup()     
        
    write_outcerts(f,newconfigs)
    
    
@activate(op_depends,op_creates)
def cross_op(func):
    """
        takes "product" of source collection parameters in computing output collections
    """
    
    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    conn = connect_to_db()
    db = conn[dbname]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]    
    ensure_indexes(db,outroots)
    ftime = FuncTime(func)      
    if func.setup:
        pass_args = func.setup()
    else:
        pass_args = {}
  
    config_generators = func.in_config_generators
    configs_list = [f() for f in config_generators]
    
    check_incerts(f,configs_list)
    
    config_product = itertools.product(*configs_list)
    
    for config_tuple in config_product:
        filenames = [get_filename(config) for config in config_tuple]
        fhs = [fs.get_version(filename) for (fs,filename) in zip(in_fs,filenames)]
        config_time = max(ftime,*[get_time(fh.update_date) for fh in fhs])
        
        flat_config = dict_union(config_tuple)
        
        if not already_exists(flat_config,out_fs,config_time):
            results =  func(in_fhs,configs,**pass_args)            
            for (fs,res) in zip(out_fs,results):
                outdata,res = interpret_res(res,flat_config)
                fs.put(res,**outdata)

    if func.cleanup:
        func.cleanup()  
        
    write_outcerts(f,config_product)


#######technical dependencies

@activate(lambda x : x[0].meta_action.__dependor__(x), lambda x : x[0].meta_action.__creator__(x))
def db_update(func,initialize,args):
    oplist = initialize(*args)
    db_ops_initialize(oplist)
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
            parents = [[op0 for op0 in oplist if ir in op0[1].outroots][0]  for ir in inroots]

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
            
            parents = [[op0 for op0 in oplist if ir in op0[1].outroots][0]  for ir in inroots]
            
            for parent in parents:
                get_op_gen(parent,oplist)
                
            func.in_config_generators = [parent[1].out_config_generator for parent in parents]
                
            def newf():
                return map(dict_union,itertools.product(*[f() for f in func.in_config_generators]))
                
            func.out_config_generator = newf
            
            
def db_ops_initialize(oplist):
    for op in oplist:
        get_op_gen(op,oplist)
        

def check_incerts(func,configs_list):
    config_stings = [get_config_string(configs) for configs in configs_list]
    incertpaths = [get_cert_path(f.dbname, root, s) for (root,s) in zip(f.inroots,config_strings)]
    incertdicts =  [cPickle.load(open(incertpath)) for incertpath in incertpaths]
    assert all([d['db'] == func.dbname and d['root'] == coll and d['configs'] == s for (coll,q,d) in zip(inroots,config_strings,incertdicts)])
   
   
def write_outcerts(func,configs):
    config_string = get_config_string(configs)
    outcertpaths = [get_cert_path(f.dbname, root, config_string) for root in f.outroots]
    for (outcertpath,outroot) in zip(outcertpaths,outroots):
        createCertificateDict(outcertpath,{'db':func.dbname, 'root':outroot, 'configs':config_string})    
        

    
#############general DB things

CERTIFICATE_ROOT = '../.db_certificates'

def initialize_certificates(creates = CERTIFICATE_ROOT):
    MakeDir(creates)

def initialize_ecc_db(creates = '../mongodb/'):
    initialize_db(creates,'ecc_db')

def initialize_db(path,name,host=None,port=None):
    os.mkdir(path)
  
    #make config
    config = {}
    config['dbpath'] = os.path.abspath(path)
    config['logpath'] = os.path.abspath(os.path.join(path,'log'))
    config['startlog'] = os.path.abspath(os.path.join(path,'startlog'))
    config['name'] = name
    
    confpath = os.path.join(path,'conf')
    F = open(confpath,'w')
    pickle.dump(config,F)
    F.close()
    
    start_db(path,host,port)
    time.sleep(10)
    conn = pymongo.Connection(host,port)
    
    db = conn['__info__']
    coll = db['__info__']
        
    coll.insert({'_id' : 'ID', 'path': path, 'name' : name},safe=True)


DB_BASE_PATH = '../mongodb/'
DATA_DB_NAME = 'data'
DATACOL_COL_NAME = '__datacols__'


def connect_to_db(depends_on = DB_BASE_PATH,host = None, port = None,verify=True):

    path = depends_on
    try:
        conn = pymongo.Connection(host,port,document_class=pm.son.SON)
    except pymongo.errors.AutoReconnect:
        start_db(path,host,port)  
        time.sleep(10)
        conn = pymongo.Connection(host,port,document_class=pm.son.SON)
    else:
        pass

    if verify:
        verify_db(conn,path)
    
    return conn
    
    
def verify_db(conn,path):
    confpath = os.path.join(path,'conf')
    
    config = pickle.load(open(confpath))
    name = config['name']
    
    if '__info__' not in conn.database_names():
        raise NoInfoDBError()
    infodb = conn['__info__']
    
    if '__info__' not in infodb.collection_names():
        raise NoInfoCollError()
    infocoll = infodb['__info__']
        
    X = infocoll.find_one({'_id' : 'ID'})
    if not X or not X.get('name') or not X.get('path'):
        raise NoIDRecError()
        
    if not X['name'] == name: 
        raise WrongNameError(name,X['name'])
    
    if not X['path'] == path:
        raise WrongPathError(path,X['path'])
    
class DBError(BaseException):
    pass    
        
class NoInputCollectionError(DBError):
    def __init__(self,incolname):
        self.msg = 'Input collection %s not found in db.' % incolname

class VerificationError(BaseException):
    pass
       
class NoInfoDBError(VerificationError):
    def __init__(self):
        self.msg = 'No __info__ database found.'
    
class NoInfoCollError(VerificationError):
    def __init__(self):
        self.msg = 'No __info__ collection found.'
 
class NoIDRecError(VerificationError):
    def __init__(self):
        self.msg = 'No ID rec found.'

class WrongNameError(VerificationError):
    def __init__(self,name,xname):
        self.msg = 'Wrong name: should be %s but is %s.' % (name,xname)
        
class WrongPathError(VerificationError):
    def __init__(self,name,xname):
        self.msg = 'Wrong path: should be %s but is %s.' % (name,xname)    
               
def start_db(path,host=None,port=None):
    confpath = os.path.join(path,'conf')
    
    config = pickle.load(open(confpath))
    
    dbpath = config['dbpath']
    logpath = config['logpath']
    startlog = config['startlog']

    optstring = '--dbpath ' + dbpath + ' --fork --logpath ' + logpath 
    
    if host != None:
        optstring += ' --bind_ip ' + host 
    if port != None:
        optstring += ' --port ' + str(port)
        
    print("DOING",'mongod ' + optstring + ' > ' + startlog)    
    os.system('mongod ' + optstring + ' > ' + startlog)

 
 
#######utils

def dict_union(dictlist):
    newdict = dictlist[0]
    for d in dictlist[1:]:
        newdict.update(d)
        
    return newdict 
    
    
def get_time(dt):
    return time.mktime(dt.timetuple()) + dt.microsecond*(10**-6)
    

def FuncTime(func):
    modulename = func.__module__
    funcname = func.__name__
    modulepath = '../' + modulename + '.py'
    fullfuncname = modulename + '.' + funcname
    Seed = [fullfuncname]
    Up = linkmanagement.UpstreamLinks(Seed)
    check = zip(Up['SourceFile'],Up['LinkSource']) + [(modulepath,funcname)]
    times = storage.ListFindMtimes(check)
    return max(times.values())


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
    if not is_instance(res,str):
        assert is_instance(res,dict) and 'summary' in res.keys() and 'res' in res.keys()
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
