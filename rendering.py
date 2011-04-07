import tempfile
import os
import itertools

from bson import SON

import numpy as np
import cairo

IMAGE_URL = 'localhost:8000/render?'


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
    return params

def renderman_render(config):
     params_list = [{'model_params':[config['image']]}]     
     tmp = tempfile.mkdtemp()
     os.chdir(tmp)
     os.system('wget ' + IMAGE_URL + 'params_list=' + json.dumps(params_list))
     zipfile = [x for x in os.listdir('.') if x.endswith('.zip')][0]
     zipname = zip[:-4]
     os.system('tar -xzvf ' + zipfile)
     imagefile = [os.path.join(zipname,x) for x in os.listdir(zipname) if x.endswith('.tif')][0]
     return open(imagefile).read()


def cairo_config_gen(args):
    ranger = lambda v : np.arange(args[v]['$gt'],args[v]['$lt'],args[v]['delta']).tolist() if isinstance(args.get(v),dict) else [args.get(v)]
    
    tx = ranger('tx')
    ty = ranger('ty')
    rxy = ranger('rxy')
    sx = ranger('sx')
    sy = ranger('sy')
    objects = args['objects']  
    patterns = args['patterns']
    width = [args['width']]
    height = [args['height']]

    param_names = ['tx','ty','rxy','sx','sy','object','pattern','width','height']
    ranges = [tx, ty, rxy, sx, sy, objects, patterns, width, height]
    params = [SON([('image' , SON(filter(lambda x: x[1] is not None, zip(param_names,p))))]) for p in itertools.product(*ranges)]  

    return params
    

def cairo_render(params,returnfh=False):
    

    height = params['height']
    width = params['width']
    
    if params.get('object'):
        object = params['object']
        pattern = params['pattern']
    
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
              
        
        render_params = SON([('objs' , [SON([('pattern' , pattern), ('segments' , object)])]),
                     ('width' , width), ('height' , height)
                    ])
        
        
        
    elif params.get('objects'):
        render_params = SON([('objs' , []),
                     ('width' , width), ('height' , height)
                    ])
        for objp in params['objects']:
            object = objp['object']
            pattern = objp['pattern']
        
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
                  
            
            render_params['objs'].append(SON([('pattern' , pattern), ('segments' , object)]))
                         
        
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
    
        actions = obj.get('actions',['fill'])
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


