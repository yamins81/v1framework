import hashlib
import zmq

NETWORK_CACHE_PORT = '5555' 
NETWORK_CACHE_TIMEOUT = 60000

def get_from_cache(obj,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    if hash in cache:
        return cache[hash]

def put_in_cache(obj,value,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    cache[hash] = value
                    

def network_cache_L(L,port=None):
    if port is None:
        port = NETWORK_CACHE_PORT
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:" + str(port))

    cache = {}
    L = set(L)
    done = set([])
    while True:
        req = sock.recv_pyobj()
        if 'get'  in req:
            if req['get'] in cache:
                sock.send_pyobj(cache[req['get']])
            else:
                sock.send_pyobj(None)
        elif 'put' in req:
            cache[req['put'][0]] = req['put'][1]
            sock.send_pyobj(True)
        else:
            sock.send_pyobj(True)
            done.add(req['DONE'])
        
        if done == L:
            break

def network_cache(port=None):
    if port is None:
        port = NETWORK_CACHE_PORT 
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:" + str(port))

    cache = {}

    while True:
        req = sock.recv_pyobj()   
        if 'get'  in req:
            try:
                is_present = req['get'] in cache
            except:
                sock.send_pyobj(False)
            else:
                if is_present:
                    sock.send_pyobj(cache[req['get']])
                else:
                    sock.send_pyobj(None)
        elif 'put' in req:
            try: 
                cache[req['put'][0]] = req['put'][1]
            except:
                sock.send_pyobj(False)
            else:
                sock.send_pyobj(True)
        else:
            sock.send_pyobj(False)

