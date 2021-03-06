import tempfile
import os
import itertools
import random
import urllib
import json

import numpy as np
import cairo

import renderer

from bson import SON

import darpa

BASE_URL = 'http://50.19.109.25'
MODEL_URL = BASE_URL + ':9999/3dmodels?'
BG_URL =  BASE_URL + ':9999/backgrounds?'

def render_image(IC,config,returnfh=False):
    generator = config['generator']
    if generator == 'renderman':
        return renderman_render(config,returnfh=returnfh)
    elif generator == 'cairo':
        return cairo_render(config,returnfh=returnfh)
    elif generator == 'darpa':
        return darpa.darpa_render(IC,config)
    else:
        raise ValueError, 'image generator not recognized'

class ImageConfigs(object):
    def __init__(self,config_gen_spec):
        self.configs = config_gen(self,config_gen_spec)
    
    def render_image(self,config,returnfh = False):
        return render_image(self,config,returnfh = returnfh)
         

def chain(iterables):
    for (ind,it) in enumerate(iterables):
        for element in it:
            yield ind,element

class config_gen(object):
    def __init__(self,IC,config):
        if not isinstance(config['images'],list):
            config['images'] = [config['images']]
        self.im_configs = config['images']
        param_list = []
        for I in config['images']:    
            if I['selection'] == 'specific':
                newparams = specific_config_gen(IC,I)
            elif I['selection'] == 'gridded':
                newparams = gridded_config_gen(IC,I)
            elif I['selection'] == 'random':
                newparams = random_config_gen(IC,I)
            elif I['generator'] == 'dataset_api':
                import dataset.compute
                DatasetObject = dataset.compute.compute_instance(I)
                meta = DataObject.columns['meta']
                files = DataObject.columns['img_fullpath']
                newparams = []
                for n,fp in zip(meta,files):
                    n['img_fullpath'] = fp
                    n['dataset_name'] = I['dataset_name']
                    newparams.append(SON([('image',n)]))
            
            param_list.append(newparams)
            
        self.param_list = chain(param_list)

            
    def __iter__(self):
        return self
        
    def next(self):
        ind,x = self.param_list.next()
        x['image']['generator'] = self.im_configs[ind]['generator']
        return x
            
        
def specific_config_gen(IC,config):
    if config['generator'] == 'darpa':
        return darpa.specific_config_gen(IC,config)
    else:
        images = config['specs']
        return [SON([('image',m)]) for m in images]  
    
def random_config_gen(IC,config):
    if config['generator'] == 'cairo':
        return cairo_random_config_gen(config)
    elif config['generator'] == 'renderman':
        return renderman_random_config_gen(config)
    elif config['generator'] == 'darpa':
        return darpa.darpa_random_config_gen(IC,config)
        
def gridded_config_gen(IC,config):
    if config['generator'] == 'cairo':
        return cairo_config_gen(config)   
    elif config['generator'] == 'renderman':
        return renderman_config_gen(config)
    elif config['generator'] == 'darpa':
        return darpa.darpa_gridded_config_gen(IC,config)

def renderman_config_gen(args):
    ranger = lambda v : np.arange(args[v]['$gt'],args[v]['$lt'],args['delta']).tolist() if isinstance(v,dict) else [args.get(v)]
    
    tx = ranger('tx')
    ty = ranger('ty')
    tz = ranger('tz')
    rxy = ranger('rxy')
    rxz = ranger('rxz')
    ryz = ranger('ryz')
    sx = ranger('sx')
    sy = ranger('sy')
    sz = ranger('sz')
    kenv = ranger('kenv')
    model_ids = args['model_ids']

    param_names = ['tx','ty','tz','rxy','rxz','ryz','sx','sy','sz','kenv','model_id']
    ranges = [tx,ty,tz,rxy,rxz,ryz,sx,sy,sz,kenv,model_ids]
    params = [SON([('image' , SON(filter(lambda x: x[1] is not None, zip(param_names,p))))]) for p in itertools.product(*ranges)]  


    chooser = lambda v : (lambda : v[random.randint(0,len(v)-1)])    
    random_ranger = lambda v : (((chooser(np.arange(v['$gt'],v['$lt'],v['delta'])) if v.get('delta') else (lambda : (v['$lt'] - v['$gt']) * random.random() + v['$gt'])))  if isinstance(v,dict) else v) if v else None
    
    if 'bg_ids' in args:
        bg_ids = args['bg_ids']
    elif 'bg_query' in args:
        bg_query = args['bg_query']
        bg_ids = json.loads(urllib.urlopen(BG_URL + 'query=' + json.dumps(bg_query) + '&distinct=path').read())
    else:
        bg_ids = None
    funcs = []
    if bg_ids:
        funcs.append(('bg_id',chooser(bg_ids)))
    if 'bg_phi' in args:
        funcs.append(('bg_phi',random_ranger(args['bg_phi'])))
    if 'bg_psi' in args:
        funcs.append(('bg_psi',random_ranger(args['bg_psi'])))
        
        
        
    for param in params:
        p = param['image']
        if args.get('use_canonical'):
            p['use_canonical'] = args['use_canonical']    
        for (k,f) in funcs:
            if f:
                p[k] = f()
        if args.get('res'):
            p['res'] = args['res']
    
    return params
    

def renderman_random_config_gen(args):
    chooser = lambda v : (lambda : v[random.randint(0,len(v)-1)])    
    ranger = lambda v : (((chooser(np.arange(v['$gt'],v['$lt'],v['delta'])) if v.get('delta') else (lambda : (v['$lt'] - v['$gt']) * random.random() + v['$gt'])))  if isinstance(v,dict) else v) if v else None
    num = args['num_images']
    funcs = [(k,ranger(args.get(k))) for k in ['tx','ty','tz','rxy','rxz','ryz','sx','sy','sz','s','bg_phi','bg_psi']]

    if not 'model_ids' in args:
        models = json.loads(urllib.urlopen(MODEL_URL + 'action=distinct&field=id').read())
    else:
        models = args['model_ids']
    funcs1 = [('model_id',chooser(models))]
    if 'bg_ids' in args:
        bg_ids = args['bg_ids']
        funcs1.append(('bg_id',chooser(bg_ids)))
    elif 'bg_query' in args:
        bg_query = args['bg_query']
        bg_ids = json.loads(urllib.urlopen(BG_URL + 'query=' + json.dumps(bg_query) + '&distinct=path').read())
        funcs1.append(('bg_id',chooser(bg_ids)))
        
    if 'kenvs' in args:
        kenvs = args['kenvs']
        funcs1.append(('kenv',chooser(kenvs)))
    
    params = []
    for i in range(num):
        p = SON([])
        if args.get('use_canonical'):
            p['use_canonical'] = args['use_canonical']    
        for (k,f) in funcs + funcs1:
            if f:
                p[k] = f()
        if args.get('res'):
            p['res'] = args['res']

        params.append(SON([('image',p)]))
        
    return params


def get_canonical_view(m):
    v = json.loads(urllib.urlopen(MODEL_URL + 'query={"id":"' + m + '"}&fields=["canonical_view"]').read())[0]
    if v.get('canonical_view'):
        return v['canonical_view']
    
    
def renderman_render(config,returnfh = False):
    config = config.to_dict()
    
    params_list = [{}]
    param = params_list[0]
    if 'bg_id' in config:
        param['bg_id'] = config.pop('bg_id')
    if 'bg_phi' in config:
        param['bg_phi'] = config.pop('bg_phi')
    if 'bg_psi' in config:
        param['bg_phi'] = config.pop('bg_psi')
    if 'kenv' in config:
        param['kenv'] = config.pop('kenv')
    if 'res' in config:
        param['res_x'] = param['res_y'] = config['res']
    use_canonical = config.pop('use_canonical',False)
    if use_canonical:
        v = get_canonical_view(config['model_id'])
        if v:
            config['rotations'] = [{'rxy':v['rxy'],'rxz':v['rxz'],'ryz':v['ryz']},
                                   {'rxy':config.pop('rxy',0),'rxz':config.pop('rxz',0),'ryz':config.pop('ryz',0)}]
    param['model_params'] = [config]   

    orig_dir = os.getcwd()
    os.chdir(os.path.join(os.environ['HOME'] , 'render_wd'))
    tmp = tempfile.mkdtemp()
    renderer.render(tmp,params_list)
    imagefile = [os.path.join(tmp,x) for x in os.listdir(tmp) if x.endswith('.tif')][0]
    os.chdir(orig_dir)
     
    fh = open(imagefile)
    if returnfh:
        return fh
    else:
        return fh.read()
    

def cairo_config_gen(args):
    ranger = lambda v : np.arange(args[v]['$gt'],args[v]['$lt'],args[v]['delta']).tolist() if isinstance(args.get(v),dict) else [args.get(v)]
    
    tx = ranger('tx')
    ty = ranger('ty')
    rxy = ranger('rxy')
    sx = ranger('sx')
    sy = ranger('sy')
    objects = args['objects']  
    patterns = args['patterns']
    action_lists = args['action_lists']
    width = [args['width']]
    height = [args['height']]

    param_names = ['tx','ty','rxy','sx','sy','object','pattern','actions','width','height']
    ranges = [tx, ty, rxy, sx, sy, objects, patterns, action_lists, width, height]
    params = [SON([('image' , SON(filter(lambda x: x[1] is not None, zip(param_names,p))))]) for p in itertools.product(*ranges)]  

    return params
    
   


def cairo_random_config_gen(args):
    chooser = lambda v : (lambda : v[random.randint(0,len(v)-1)])    
    ranger = lambda v : (((chooser(np.arange(v['$gt'],v['$lt'],v['delta'])) if v.get('delta') else (lambda : (v['$lt'] - v['$gt']) * random.random() + v['$gt'])))  if isinstance(v,dict) else v) if v else None
    num = args['num_images']
    funcs = [(k,ranger(args.get(k))) for k in ['tx','ty','rxy','sx','sy']]
    funcs1 = [(k, chooser(args[k[0]])) for k in [('objects','object'),
                                                 ('patterns','pattern'),
                                                 ('action_lists','actions')] if k[0] in args]
    params = []
    for i in range(num):
        param = SON([])
        param['height'] = args['height']
        param['width'] = args['width']
        for (k,f) in funcs:
            if f:
                param[k] = f()
        for (k,f) in funcs1:
            param[k[1]] = f()
        params.append(SON([('image',param)]))
        
    return params

def cairo_render(params,returnfh=False):
    

    height = params['height']
    width = params['width']
    
    if params.get('object'):
        object = params['object']
        pattern = params['pattern']
        actions = params.get('actions',['fill'])
    
        if params.get('sx') or params.get('sy'):
            S = SON([('type','scale'),('args',(params.get('sx',1),params.get('sy',1)))])
            InvS = SON([('type','scale'),('args',(1./params.get('sx',1),1./params.get('sy',1)))])
            object = [S] + object + [InvS]
    
     
        if params.get('rxy'):
            Rot = SON([('type','rotate'),('args',(params.get('rxy'),))]) 
            InvRot = SON([('type','rotate'),('args',(-params.get('rxy'),))])
            object = [Rot] + object + [InvRot]    
    
        if params.get('tx') != None or params.get('ty') != None:
            Tr = SON([('type','translate'),('args',(params.get('tx',0),-params.get('ty',0)))])
            InvTr = SON([('type','translate'),('args',(-params.get('tx',0),params.get('ty',0)))])
            object = [Tr] + object + [InvTr]
           
            
        
        Tr = SON([('type','translate'),('args',(.5,.5))])
        InvTr = SON([('type','translate'),('args',(-.5,-.5))])
        object = [Tr] + object + [InvTr]
              
        
        render_params = SON([('objs' , [SON([('pattern' , pattern), ('segments' , object) , ('actions',actions)])]),
                     ('width' , width), ('height' , height)
                    ])
        
        
        
    elif params.get('objects'):
        render_params = SON([('objs' , []),
                     ('width' , width), ('height' , height)
                    ])
        for objp in params['objects']:
            object = objp['object']
            pattern = objp['pattern']
            actions = objp.get('actions',['fill'])
        
            if objp.get('sx') or objp.get('sy'):
                S = SON([('type','scale'),('args',(objp.get('sx',1),objp.get('sy',1)))])
                InvS = SON([('type','scale'),('args',(1./objp.get('sx',1),1./objp.get('sy',1)))])
                object = [S] + object + [InvS]
        
         
            if objp.get('rxy'):
                Rot = SON([('type','rotate'),('args',(objp.get('rxy'),))]) 
                InvRot = SON([('type','rotate'),('args',(-objp.get('rxy'),))])
                object = [Rot] + object + [InvRot]    
        
            if objp.get('tx') != None or objp.get('ty') != None:
                Tr = SON([('type','translate'),('args',(objp.get('tx',0),-objp.get('ty',0)))])
                InvTr = SON([('type','translate'),('args',(-objp.get('tx',0),objp.get('ty',0)))])
                object = [Tr] + object + [InvTr]
               
                
            
            Tr = SON([('type','translate'),('args',(.5,.5))])
            InvTr = SON([('type','translate'),('args',(-.5,-.5))])
            object = [Tr] + object + [InvTr]
                  
            
            render_params['objs'].append(SON([('pattern' , pattern), ('segments' , object), ('actions',actions)]))
                         
        
    return cairo_render_image(render_params,returnfh=returnfh)                     


def cairo_render_image(params,returnfh=False):

    WIDTH, HEIGHT = params['width'], params['height']
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)

    ctx.scale (WIDTH, HEIGHT) # Normalizing the canvas
  
    objs = params['objs']
    for obj in objs:
    
        pattern = obj.get('pattern')
        if pattern:
            pat_type = pattern['type']
            pat_args = pattern['args']
            pat_stops = pattern.get('color_stops',[])
            pattern = getattr(cairo,pat_type)(*pat_args)
            for stop in pat_stops:
                stop_type = stop['type']
                stop_args = stop['args']
                getattr(pattern,'add_color_stop_' + stop_type)(*stop_args)
        
            ctx.set_source(pattern)
   
        segments = obj['segments']
        
        for segment in segments:
            segment_type = segment['type']
            segment_args = segment.get('args',())
            getattr(ctx,segment_type)(*segment_args)
    
        actions = obj['actions']
        for action in actions:  
            getattr(ctx,action)()

    fh = tempfile.TemporaryFile()
    
    surface.write_to_png(fh) # Output to PNG
    
    fh.seek(0)
    
    if returnfh:
        return fh
    else:
        s = fh.read()
        return s


