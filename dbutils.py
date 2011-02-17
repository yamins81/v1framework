from collections import OrderedDict

#------ ops


def process_query(func,query):

    inroots = func.inroots

    if query is not None:
        if isinstance(query,OrderedDict):
 		    query = [query]*len(inroots)
	    else:
		    assert isinstance(query,list) and len(query) == len(inroots) and all([isinstance(q,OrderedDict) for q in query])
    else:
	    query = [OrderedDict([])]*len(inroots)
		
   return query


def get_cert_path(dbname,root,query):
    pass
	
def op_depends(x):
    f = x[0]
    query = x['query']
    query = process_query(f,query)
    deps = [get_cert_path(f.dbname, r, q) for (q,r) in zip(f.inroots,query)]
    return tuple(deps)
    
    
def op_creates(x):
    f = x[0]  
    query = x['query']
    query = process_query(f,query)
    params = x['params']
    query.update(params)
    creates = [get_cert_path(f.dbname, root, query) for root in f.outroots]
    return tuple(creates)


def check_incerts(func,query):
    incertpaths = op_depends({0:func,'query':query}) 
    query = process_query(f,query)
    incertdicts =  [cPickle.load(open(incertpath)) for incertpath in incertpaths]
    assert all([d['db'] == func.dbname and d['root'] == coll and d['query'] == q for (coll,q,d) in zip(inroots,query,incertdicts)])
    
    
def write_outcerts(func,query,params=None):
    if params is None:
        params = {}
    outcertpaths = op_creates({0:func,'query':query,'params':params}) 
    query = process_query(f,query)
    out_query = query[0]
    for k in query[1:]:
        out_query.update(k)
    out_query.update(params)
    for (outcertpath,outroot) in zip(outcertpaths,outroots)
        createCertificateDict(outcertpath,{'db':func.dbname, 'root':outroot, 'query':out_query})    
   

def reach_in(attr,q):
    q1 = OrderedDict([])
    for k in q:
        q1[attr + '.' + k] = q[k]
    return q1   
    
def ensure_indexes(db,roots):
    colls = [db[coll + '.files'] for coll in roots]
    for coll in colls:
        coll.ensure_index('config',unique=True)
 
def do_rec(in_fs,func):
    if in_fs:
    	in_fhs = [fs.get(id) for (id,fs) in zip(ids,in_fs)]
    else:
        in_fhs = None
        
	if func.caches:
		cache_data = []
		cache_configs = []
		cache_fhs = []
		for c_coll, c_func, c_fs in zip(func.cache_colls,func.cache_funcs,func.cache_fs):
			c_config = c_func(config,param_names)
			cache_configs.append(c_config)
			c_data = c_coll.find_one(reach_in('config',c_config))
			cache_data.append(c_data)
			if c_data:
			    c_fs = c_fs.get(c_data['_id'])
			    cache_fhs.append(c_fs)
	
	    if in_fhs:
    		results, new_cache_data = func(in_fhs,config,cache_fhs)	
    	else:
    	    results, new_cahce_data = func(config,cache_fhs)	
		
		for (c_fs,new_c_data,c_config) in zip(func.cache_fs,new_cache_data,cache_configs):
			if new_c_data:
				c_fs.put(new_c_data,{'config':c_config})
				
	else:  
	    if in_fhs:
    		results = func(in_fhs,config)
    	else:
    	    results = func(config)
	
	return results
 
def prepare_func_caches(func):
    if func.caches:
        if is_str_like(func.caches):
            func.caches = [func.caches]
        assert isinstance(func.caches,list)
        for (i,c) in enumerate(func.caches):
            if is_str_like(c):
                func.caches[i] = (c,lambda x : x)
            else:
                assert isinstance(c,tuple) and len(c) == 2
            
        cache_roots, cache_funcs =  zip(*func.caches)
        func.cache_roots = cache_roots
        func.cache_func = cache_funcs
        func.cache_colls = [db[root + '.files'] for root in func.cache_roots]
        func.cache_fs = [gridfs.GridFS(db,collection = coll) for coll in func.cache_roots]


def interpret_res(res,config):
    outdata = {'config' : config.copy()}
    
	if not is_instance(res,str):
		assert is_instance(res,dict) and 'summary' in res.keys() and 'res' in res.keys()
		summary = res['summary']
		res = res['res']
		outdata['config'].update(summary)
	
	return outdata,res

     
def already_exists(config,fs_list):

    exists = [fs.exists(reach_in('config',config)) for fs in fs_list]
    
    assert (not any(exists)) or all(exists)
    
    return all(exists)
 
 
@creates(op_creates)
def inject_op(func, query=None):

    query = process_query(func,query)
    outroots = func.outroots
    dbname = func.dbname
    conn = connect_to_db()
    db = conn[dbname]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    ensure_indexes(db,outroots)
    prepare_func_caches(func)
    
    configs = func.generate_configs(query)
              
    for config in configs:
        assert isinstance(config,OrderedDict)
        if not already_exists(config,out_fs):
			results = do_rec(None,func)
	
			for (fs,res) in zip(out_fs,results):
				outdata,res = interpret_res(res,config)
				fs.put(res,**outdata)
				
                         
    write_outcerts(func,query)
            

@activate(op_depends,op_creates)
def cross_op(func,query = None,params=None):

    check_incerts(func,query)
    query = process_query(func,query)
    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    innames = [inroot + '.files' for inroot in inroots]
    conn = connect_to_db()
    db = conn[dbname]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]    
    ensure_indexes(db,outroots)
    prepare_func_caches(func)
    if params is None:
        params = {}
  
    cursors = [db[coll].find(reach_in('config',q)) for (q,coll) in zip(query,innames)]
    
    I = itertools.product(cursors)

    for T in I:
        ids = [t.pop('_id') for t in T]
        
        config = T[0]['config']
        param_names = [T[0]['config'].keys()]
        for t in T[1:]:
            config.update(t['config'])
            param_names.append(t['config'].keys())
        config.update(params)  
        param_names.append(params.keys())
                    
        if not already_exists(config,out_fs):

            results = do_rec(in_fs,func)
			
			for (fs,res) in zip(out_fs,results):
			    outdata,res = interpret_res(res,config)
				fs.put(res,**outdata)
        
    write_outcerts(func,query,params=params)

	
@activate(op_depends,op_creates)
def dot_op(func, query = None, params=None):

    check_incerts(func,query)
    query = process_query(func,query)
    inroots = func.inroots
    outroots = func.outroots
    dbname = func.dbname
    innames = [inroot + '.files' for inroot in inroots]
    conn = connect_to_db()
    db = conn[dbname]
    in_fs = [gridfs.GridFS(db,collection = coll) for coll in inroots]
    out_fs = [gridfs.GridFS(db,collection = coll) for coll in outroots]
    ensure_indexes(db,outroots)
    prepare_func_caches(func)
    if params is None:
        params = {}

    cursors = [db[coll].find(reach_in('config',q)).sort('config') for (q,coll) in zip(query,innames)]
    
    I = itertools.izip(cursors)
        
    for T in I:     
        ids = [t.pop('_id') for t in T]
        
        configs = [t['config'] for t in T]
        assert([c == configs[0] for c in configs]) 
        config = configs[0]
        config.update(params)
    
        if not already_exists(config,out_fs):
			results = do_rec(in_fs,func)
			
			for (fs,res) in zip(out_fs,results):
			    outdata,res = interpret_res(res,config)
				outdata['_id'] = ids[0]
				fs.put(res,**outdata)
        
    write_outcerts(func,query,params=params)
     


            
@depends_on(lambda x : x[0].meta_action.__dependor__(x), lambda x : x[0].meta_action.__creator__(x))
def db_update(func,query):    
    meta_action = func.meta_action  
    meta_action(func,query)



###########------ decorators

def inject(dbname,outroots,generator):
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.meta_action = inject_op
        f.generate_configs = generator
        f.dbname = dbname
        f.inroots = ['']
        f.outroots = outroots
        return f
    return func
    
    
def dot(dbname,inroots,outroots):
    if is_string_like(inroots):
        inroots = [inroots]
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.meta_action = dot_op
        f.inroots = inroots
        f.outroots = outroots
        return f
    return func
    
def cross(dbname,inroots,outroots,caches=None):
    if is_string_like(inroots):
        inroots = [inroots]
    if is_string_like(outroots):
        outroots = [outroots]
    def func(f):
        f.meta_action = cross_op
        f.inroots = inroots
        f.outroots = outroots
        f.caches = caches
        return f
    return func
        
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


    
import inspect, sys

def check_unique_ops(level = 1):
    F = sys._getframe(level)
    L = F.f_globals
    
    O = []
    B = {}
    for l in L:
        obj = L[l]
        if inspect.isroutine(obj) and hasattr(obj,'outroots'):
            outroots = obj.outroots
            O.extend(outroots)
            for outroot in outroots:
                if not outroot in B:
                    B[outroot] = []
                B[outroot].append(obj.__name__)
                    
            
    Oset = set(O)
    bad = []
    for o in Oset:
        c = O.count(o)
        if c > 1:
            bad.append(o)
    
    if bad:
        raise NotUniqueError(bad,B)


class NotUniqueError(Exception):
    
    def __init__(self,bad,B):
        badfuncs = [B[b] for b in bad]
        self.msg = 'Bad values ' + repr(bad) + ' for ' + repr(badfuncs) 
