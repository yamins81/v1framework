import os
import copy
import numpy as np
import scipy as sp
import Image
import ImageOps
import tabular as tb
import pythor3.wildwest.bbox as bbox

from bson import SON

xfields = ['BoundingBox_X1', 'BoundingBox_X2',  'BoundingBox_X3','BoundingBox_X4']
yfields = ['BoundingBox_Y1', 'BoundingBox_Y2',  'BoundingBox_Y3','BoundingBox_Y4']
otherfields = ['ObjectType','Occlusion','Ambiguous','Confidence']

def uniqify(X):
    return [x for (i,x) in enumerate(X) if x not in X[:i]]

def darpa_image_path(t):
    return t['clip_num'] + '_' + str(t['Frame']) + '.jpg'

def darpa_random_config_gen(IC,args):
    IC.current_frame_path = None
    IC.base_dir = args['base_dir']
    mdp = os.path.join(IC.base_dir,'__metadata__.csv')
    IC.metadata = X = tb.tabarray(SVfile = mdp)
    IC.num_images = args['num_images']
    IC.size = args['size']
    T = np.unique(X[['clip_num','Frame']])
    im_stuff = {}
    params = []
    for i in range(IC.num_images):
        print('At image', i)
        ind = np.random.randint(len(T))
        t = T[ind]
        clip_num = t['clip_num']
        frame = t['Frame']
        p = darpa_image_path(t)
        add_im_stuff(im_stuff,IC,p,t)
        box = choose_random_darpa_box(im_stuff[p]['size'],IC.size)
        intersects_with = get_darpa_intersection(box,im_stuff[p]['boxes'])
        for (iwind,iw) in enumerate(intersects_with):
             iw = copy.deepcopy(iw)
             b = iw.pop('box')
             iw['bounding_box'] = SON([('xfields',b.xs),('yfields',b.ys)])
             intersects_with[iwind] = iw
        label = uniqify([iw['ObjectType'] for iw in intersects_with])
        label.sort()
        p = SON([('size',IC.size),         
                 ('bounding_box',SON([('xfields',box.xs),('yfields',box.ys)])),
                 ('intersects_with',intersects_with),
                 ('ObjectType',label),
                 ('clip_num',clip_num),
                 ('Frame',frame),
                 ('base_dir',IC.base_dir)])
        p = SON([('image',p)])
        params.append(p)

    if args.get('enrich_positives',False):
        for x in X:
            p = darpa_image_path(x)
            print(p)
            add_im_stuff(im_stuff,IC,p,x,get_boxes=False)
            box = bbox.BoundingBox(xs = [x[xf] for xf in xfields],
                                   ys = [x[yf] for yf in yfields])
            xc,yc = box.center
            center = correct_center((xc,yc),IC.size,im_stuff[p]['size'])
            box = bbox.BoundingBox(center = center,width = IC.size[0], height = IC.size[1])
            label = x['ObjectType']
            p = SON([('size',IC.size),
                     ('bounding_box',SON([('xfields',box.xs),('yfields',box.ys)])),
                     ('ObjectType',label),
                     ('clip_num',x['clip_num']),
                     ('Frame',x['Frame']),
                     ('base_dir',IC.base_dir),
                     ('enriched',True)])
            params.append(SON([('image',p)]))
            
    js = np.array( [p['image']['clip_num'] + '_' + str(p['image']['Frame']) for p in params])
    js_ag = js.argsort()
    params = [params[ind] for ind in js_ag]
    return params

def correct_center(center,shp,size):
    (xc,yc) = center
    xc,yc = (int(round(xc)),int(round(yc)))
    (w,h) = shp
    width,height = size
    w0 = w/2 ; w1 = w - w0
    h0 = h/2 ; h1 = h - h0

    dx = max(0,w0-xc) + min(width - xc-w1,0)
    dy = max(0,h0-yc) + min(height - yc-h1,0)

    xc = xc + dx
    yc = yc + dy

    return xc,yc
    

def add_im_stuff(im_stuff,IC,p,t,get_boxes = True):
    clip_num = t['clip_num']
    frame = t['Frame']
    if p not in im_stuff:
        path = os.path.join(IC.base_dir,darpa_image_path(t))
        Im = Image.open(path)
        if get_boxes:
            all_boxes = get_all_darpa_boxes(IC.metadata,clip_num,frame)
            im_stuff[p] = {'size':Im.size,'boxes':all_boxes}
        else:
            im_stuff[p] = {'size':Im.size}

import StringIO

def darpa_render(IC,config):
    path = os.path.join(IC.base_dir,darpa_image_path(config))
    if IC.current_frame_path != path:
        IC.current_frame_path = path
        IC.current_frame = Image.open(path)
    xs = config['bounding_box']['xfields']
    ys = config['bounding_box']['yfields']
    box = (xs[0],ys[0],xs[2],ys[1])
    im = IC.current_frame.crop(box)

    f = StringIO.StringIO()
    im.save(f, "JPEG")
    data = f.getvalue()
    return data

def choose_random_darpa_box(im_size,size):
    assert im_size[0] >= size[0]
    assert im_size[1] >= size[1]
    sx = np.random.randint(im_size[0]-size[0])
    sy = np.random.randint(im_size[1]-size[1])
    box = bbox.BoundingBox(xs = (sx,sx+size[0],sx+size[0],sx),
                           ys = (sy,sy,sy+size[1],sy+size[1]))

    return box

def get_all_darpa_boxes(X,cn,fr):
    X = X[(X['clip_num'] == cn) & (X['Frame'] == fr)]
    boxes = []
    for x in X:
        box = bbox.BoundingBox(xs = [x[xf] for xf in xfields],
                               ys = [x[yf] for yf in yfields])
        obj = SON([('box',box)] + [(of,x[of]) for of in otherfields])
        boxes.append(obj)
    return boxes

def get_darpa_intersection(box,boxes):
    intersects_with = []
    for box2 in boxes:
        box2r = box2['box']
        au = box | box2r
        ai = box & box2r
        if ai / au >= .2:
            intersects_with.append(box2)
    return intersects_with
        

def get_random_empty_bbox(metadata,sizes,imagedir):
    try_num = 0
    while True:
        shp = sizes[sp.random.randint(0,high=len(sizes))]
        random_row = metadata[np.random.randint(len(metadata))]
        clip_num = random_row['clip_num']
        frame = random_row['Frame']
        fl = os.path.join(imagedir,clip_num + '_' + str(frame) + '.jpg')
        im = get_image(fl)
        try_num+= 1
        sy,sx = im.shape
        if sx >= shp[0] and sy >= shp[1]:
            start_x = np.random.randint(sx - shp[0])
            start_y = np.random.randint(sy - shp[1])
            print('trying',clip_num,frame,start_x,start_y)
            if no_intersection(start_x,start_y,shp,metadata,clip_num,frame):
                break
    
    return im[start_y : start_y + shp[1] , start_x : start_x + shp[0]]


def no_intersection(sx,sy,shp,metadata,cn,fr):
    M = metadata[(metadata['clip_num'] == cn) & (metadata['Frame'] == fr)]
    box1 = ((sx,sy),(sx+shp[0],sy),(sx+shp[0],sy+shp[1]),(sx,sy+shp[1]))
    for obj in M:
        b2x = [obj[xf] for xf in xfields]
        b2y = [obj[yf] for yf in yfields]
        box2 = zip(b2x,b2y)
        if box_intersection(box1,box2):
            return False
        
    return True

def box_in(b1,b2):
    ((x10,y10),(x11,y10),(x11,y11),(x10,y11)) = b1
    ((x20,y20),(x21,y20),(x21,y21),(x20,y21)) = b2

    x1min = min(x10,x11) ; x1max = max(x10,x11)
    y1min = min(y10,y11) ; y1max = max(y10,y11)
    x2min = min(x20,x21) ; x2max = max(x20,x21)
    y2min = min(y20,y21) ; y2max = max(y20,y21)
    
    return x1min <= x2min and x1max >= x2max and y1max >= y2max and y1min <= y2min
              
def box_intersection(b1,b2):
    if not (box_in(b1,b2) or box_in(b2,b1)):
        lines1 = lines_from_box(b1)
        lines2 = lines_from_box(b2)
        for l1 in lines1:
            for l2 in lines2:
                if line_intersection(l1,l2):
                    return True
    if box_in(b1,b2) or box_in(b2,b1):
        return True

    return False
    
def lines_from_box(box):
    return [(box[0],box[1]),(box[1],box[2]),(box[2],box[3]),(box[3],box[0])]

def line_intersection(l0,l1):
    m0,b0 = get_mb(l0)
    m1,b1 = get_mb(l1)
    if not (np.isinf(m0) or np.isinf(m1)):
        l0xmin = min(l0[0][0],l0[1][0])
        l0xmax = max(l0[0][0],l0[1][0])
        l1xmin = min(l1[0][0],l1[1][0])
        l1xmax = max(l1[0][0],l1[1][0])
        if m1 != m0:
            xint = (b0 - b1)/(m1 - m0)
            return (l0xmin <= xint <= l0xmax) and (l1xmin <= xint <= l1xmax)
        else:
            if b0 != b1:
                return False
            else:
                not (l0xmax < l1xmin or l0xmin >= l1xmax)
    else:
        if (not np.isinf(m0)) and np.isinf(m1):
            return inf_line_intersection(l0,l1,m0,b0)
        elif (not np.isinf(m1)) and np.isinf(m0):
            return inf_line_intersection(l1,l0,m1,b1)
        else:
            return l0[0][0] == l1[0][0]

def inf_line_intersection(l0,l1,m0,b0):
    l1x = l1[0][0]
    l1ymin = min(l1[0][1],l1[1][1])
    l1ymax = max(l1[0][1],l1[1][1])
    yint = m0*l1x + b0
    return (l1ymin <= yint <= l1ymax)


def get_mb(l):
    ((x0,y0),(x1,y1)) = l

    if x0 != x1:
        m = (y0 - y1) / (x0 - x1)
        b = (y0 + y1 - m*(x0 +x1))/2
    else:
        m = np.inf
        b = None
        
    return m,b
            

    
                
