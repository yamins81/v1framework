import tempfile
import os
import itertools

from collections import OrderedDict

import numpy as np
import cairo

IMAGE_URL = 'localhost:8000/render?'

ranger = lambda v : np.arange(args[v]['$gt'],args[v]['$lt'],args['delta']).tolist() if isinstance(v,dict) else [args.get(v)]

def renderman_config_gen(args):
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

    param_names = ['tx','ty','tz','rxy','rxz','ryz','sx','sy','sz','kenv','model_id','generator']
    ranges = [tz,ty,tz,rxy,rxz,ryz,sx,sy,sz,kenv,model_ids,generator]
    params = [OrderedDict([('image' , OrderedDict(filter(lambda x: x[1], zip(param_names,p))))]) for p in itertools.product(*ranges)]  
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
    tx = ranger('tx')
    ty = ranger('ty')
    rxy = ranger('rxy')
    sx = ranger('sx')
    sy = ranger('sy')
    objects = args['objects']  
    patterns = args['patterns']
    width = [args['width']]
    height = [args['height']]

    param_names = ['tx','ty','rxy','sx','sy','object','pattern','width','height','generator']
    ranges = [tz,ty,rxy,sx,sy,objects,patterns,width,height , generator]
    params = [OrderedDict([('image' , OrderedDict(filter(lambda x: x[1], zip(param_names,p))))]) for p in itertools.product(*ranges)]  
    return params
    

def cairo_render(params):
    
    object = params['object']
    pattern = params['pattern']
    height = params['height']
    width = params['width']
    
    if params.get('sx') or params.get('sy'):
        S = OrderedDict([('type','scale'),('args',(params.get('sx',1),params.get('sy',1)))])
        InvS = OrderedDict([('type','scale'),('args',(1./params.get('sx',1),1./params.get('sy',1)))])
        object = [S] + object + [InvS]
    
    if params.get('rxy'):
        Rot = OrderedDict([('type','rotate'),('args',(params.get['rxy'],))]) 
        InvRot = OrderedDict([('type','rotate'),('args',(params.get['rxy'],))])
        object = [Rot] + object + [InvRot]
        
    if params.get('tx') != None or params.get('ty') != None:
        Tr = OrderedDict([('type','translate'),('args',(params.get('tx',0),params.get('ty',0)))])
        InvTr = OrderedDict([('type','translate'),('args',(-params.get('tx',0),-params.get('ty',0)))])
        object = [Tr] + obejct + [InvTr]
        
    
    params = OrderedDict([('objs' , [OrderedDict([('pattern' , pattern), ('segments' , object)])]),
                 ('width' , width), ('height' , height)
                ])
                
    return cairo_render_image(params)


def cairo_render_image(params):

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
    s = fh.read()
    return s


