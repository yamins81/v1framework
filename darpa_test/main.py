import os
import numpy as np
import scipy as sp
import Image
import ImageOps
import tabular as tb
import pythor3.wildwest.bbox as bbox

import filter_generation as fg
from processing import image2array, preprocess, postprocess

from pythor_protocols import fbcorr, old_norm, lnorm, lpool, c_numpy_mixed, transform_average

from scikits.learn import linear_model
import scikits.learn.svm as svm

def get_image(file):
    im = ImageOps.grayscale(Image.open(file))
    return sp.misc.fromimage(im)

xfields = ['BoundingBox_X1', 'BoundingBox_X2',  'BoundingBox_X3','BoundingBox_X4']
yfields = ['BoundingBox_Y1', 'BoundingBox_Y2',  'BoundingBox_Y3','BoundingBox_Y4']
        
def detect_train_and_evaluate(model,metadatafile,imagedir,train_frames,test_frames,num_empties=None,regress=True):
    clf = detect_train(model,metadatafile,imagedir,train_frames,num_empties=num_empties,regress=regress)
    predictions = detect_evaluate(model,imagedir,clf,test_frames)
    return clf,predictions

def check_stamps(metadatafile,imagedir,train_frames,outdir):
    os.mkdir(outdir)
    metadata = tb.tabarray(SVfile=metadatafile)
    #get labels for training objects

    train_labels_inds = []
    for cn,fr in train_frames:
        inds = ((metadata['Frame'] == fr) & (metadata['clip_num'] == cn) & (metadata['ObjectType'] != 'DCR')).nonzero()[0]
        #ind = inds[t['object_number']]
        train_labels_inds.extend(inds)
    train_labels = metadata[train_labels_inds]
    #get stamps for training objects

    train_points = []
    train_points_labels = []
    sizes = []
    num_train = 0
    for label in train_labels:
        lbl = label['clip_num'] + '_' + str(label['Frame']) + '.jpg'
        print(label)
        framefile = os.path.join(imagedir,lbl)
        im = get_image(framefile)

        box = bbox.BoundingBox(xs = [label[xf] for xf in xfields],
                               ys = [label[yf] for yf in yfields])
        stamp = bbox.stamp(im,box,stamp_shape=(200,200))[0]
        if stamp is not None:
            img = Image.fromarray(stamp)
            img.save(os.path.join(outdir,str(num_train) + '.jpg'))
            num_train += 1

def detect_train(model,metadatafile,imagedir,train_frames,num_empties=None,regress=True,points=True,pset=None,stamp_shape=0):
    if points is False:
        assert regress == False, 'regress must be false if points is False'
        assert pset is not None, 'pset must not be nont if points is False'
        assert stamp_shape > 0
        transform_config = {'transform_name':'translation','percentile':pset}

    filters = fg.get_hierarchical_filterbanks(model['config']['model']['layers'])
    
    metadata = tb.tabarray(SVfile=metadatafile)
    #get labels for training objects

    train_labels_inds = []
    for cn,fr in train_frames:
        inds = ((metadata['Frame'] == fr) & (metadata['clip_num'] == cn) & (metadata['ObjectType'] != 'DCR')).nonzero()[0]
        #ind = inds[t['object_number']]
        train_labels_inds.extend(inds)
    train_labels = metadata[train_labels_inds]
    #get stamps for training objects

    train_points = []
    train_points_labels = []
    sizes = []
    num_train = 0
    for label in train_labels:
        lbl = label['clip_num'] + '_' + str(label['Frame']) + '.jpg'
        print(label)
        framefile = os.path.join(imagedir,lbl)
        im = get_image(framefile)
        box = bbox.BoundingBox(xs = [label[xf] for xf in xfields],
                               ys = [label[yf] for yf in yfields])
        stamp = bbox.stamp(im,box,stamp_shape=stamp_shape)[0]
        if stamp is not None:
            sizes.append(stamp.shape)
            print(stamp.shape)
            try:
                features = get_features(model,filters,stamp)
            except:
                print(label,'is bad')
            else:
                num_train += 1
                if points:
                    feature_points,sh = get_feature_points(features)
                    train_points.extend(feature_points)
                    vecs = get_positions((box.width,box.height),features,regress=regress)
                    train_points_labels.extend(vecs)
                else:
                    features = {'0':features}
                    feature_stats = transform_average(features,transform_config,model)
                    train_points.append(feature_stats)
                    train_points_labels.append(1)

    num_empties = (num_empties is not None) or num_train
    for ind in range(num_empties):
        print('empty',ind)
        im = get_random_empty_bbox(train_labels,sizes,imagedir)
        try:
            features = get_features(model,filters,im)
        except:
            print('empty', ind, 'is bad')
        else:
            if points:
                feature_points,sh = get_feature_points(features)
                train_points.extend(feature_points)
                if regress:
                    vecs = [(-100,-100) for ind in range(len(feature_points))]
                else:
                    vecs = [0 for ind in range(len(feature_points))]
                train_points_labels.extend(vecs)
            else:
                features = {'0':features}
                feature_stats = transform_average(features,transform_config,model)
                train_points.append(feature_stats)
                train_points_labels.append(0)

    train_points = np.array(train_points)
    train_points_labels = np.array(train_points_labels)

    #run regression
    if regress:
        clf = linear_model.LinearRegression()
    else:
        clf = svm.LinearSVC()
    clf.fit(train_points,train_points_labels)

    return clf

def detect_evaluate(model,imagedir,clf,test_frames,slice=None):
    filters = fg.get_hierarchical_filterbanks(model['config']['model']['layers'])
    #extract features from test frames
    predictions = []
    for clip_num,frame in test_frames:
        framefile = os.path.join(imagedir,clip_num + '_' + str(frame) + '.jpg')
        im = get_image(framefile)
        if slice is not None:
            im = im[slice]
        tf = get_features(model,filters,im)
        
        test_points,sh = get_feature_points(tf)
        res = clf.predict(test_points)
        predictions.append(res)

    return predictions

def detect_evaluate_spots(model,imagedir,train_frames,metadatafile,clf,stamp_shape,pset,num_empties=None):
    filters = fg.get_hierarchical_filterbanks(model['config']['model']['layers'])
    
    metadata = tb.tabarray(SVfile=metadatafile)
    #get labels for training objects
    transform_config = {'transform_name':'translation','percentile':pset}
    train_labels_inds = []
    for cn,fr in train_frames:
        inds = ((metadata['Frame'] == fr) & (metadata['clip_num'] == cn) & (metadata['ObjectType'] != 'DCR')).nonzero()[0]
        #ind = inds[t['object_number']]
        train_labels_inds.extend(inds)
    train_labels = metadata[train_labels_inds]
    #get stamps for training objects

    train_points = []
    train_points_labels = []
    sizes = []
    num_train = 0
    for label in train_labels:
        lbl = label['clip_num'] + '_' + str(label['Frame']) + '.jpg'
        print(label)
        framefile = os.path.join(imagedir,lbl)
        im = get_image(framefile)
        box = bbox.BoundingBox(xs = [label[xf] for xf in xfields],
                               ys = [label[yf] for yf in yfields])
        stamp = bbox.stamp(im,box,stamp_shape=stamp_shape)[0]
        if stamp is not None:
            sizes.append(stamp.shape)
            print(stamp.shape)
            try:
                features = get_features(model,filters,stamp)
            except:
                print(label,'is bad')
            else:
                num_train += 1
                features = {'0':features}
                feature_stats = transform_average(features,transform_config,model)
                train_points.append(feature_stats)
                train_points_labels.append(1)

    num_empties = (num_empties is not None) or num_train
    for ind in range(num_empties):
        print('empty',ind)
        im = get_random_empty_bbox(train_labels,sizes,imagedir)
        try:
            features = get_features(model,filters,im)
        except:
            print('empty', ind, 'is bad')
        else:
            features = {'0':features}
            feature_stats = transform_average(features,transform_config,model)
            train_points.append(feature_stats)
            train_points_labels.append(0)

    train_points = np.array(train_points)
    train_points_labels = np.array(train_points_labels)

    prediction = clf.predict(train_points)
    return prediction,train_points_labels


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
            

def get_features(model_config, filters, array):
    arr_dict = compute_features(model_config,filters,array)
    for k in arr_dict:
        arr_dict[k] = arr_dict[k][0]
    return arr_dict

def compute_features(model_config, filters, array):
    m_config = model_config['config']['model']

    convolve_func = c_numpy_mixed
    
    if isinstance(m_config,list):
        reslist = []
        for (filt,m) in zip(filters,m_config):
            image_fh.seek(0)
            res = compute_features_core(image_fh,filt,{'config':{'model':m}},convolve_func)
            reslist.append(res)
        return reslist
    else:
        conv_mode = m_config['conv_mode']    
        layers = m_config['layers']
        feed_up = m_config.get('feed_up',False)
        
        array,orig_imga = preprocess(array,m_config)
#        for k in array:
#            array[k] = array[k].reshape(array[k].shape + (1,))
        assert len(filters) == len(layers)
        dtype = array[0].dtype
        
        array_dict = {}
        for (ind,(filter,layer)) in enumerate(zip(filters,layers)):
            if feed_up:
                array_dict[ind-1] = array
            print(array[0].shape,'filter')
            if filter is not None:
                array = fbcorr(array, filter, layer , convolve_func)
  
            print(array[0].shape,'pool')
            if layer.get('lpool'):
                array = lpool(array,conv_mode,layer['lpool'])

            print(array[0].shape,'lnorm')
            if layer.get('lnorm'):
                if layer['lnorm'].get('use_old',False):
                    array = old_norm(array,conv_mode,layer['lnorm'])
                else:
                     array = lnorm(array,conv_mode,layer['lnorm'])
            print(array[0].shape)
    

        array_dict[len(layers)-1] = array
            
        return array_dict

    

def get_positions(s,tf,regress=True):
    layers = tf.values()
    layer_shapes = [layer.shape for layer in layers]
    ls0 = max(layer_shapes)

    #ls0 = layer_shapes[0]
    #assert all([ls[:2] == ls0[:2] for ls in layer_shapes])    
    
    xpos = (np.arange(ls0[0])/np.float(ls0[0]))
    ypos = (np.arange(ls0[1])/np.float(ls0[1]))

    if regress:
        return [(x,y) for y in ypos for x in xpos]
    else:
        return [1 for y in ypos for x in xpos]
    

def get_feature_points(tf):
    layers = tf.values()
    layer_shapes = [layer.shape for layer in layers]
#    ls0 = layer_shapes[0]
#    assert all([ls[:2] == ls0[:2] for ls in layer_shapes])
    ls0 = (max([ls[0] for ls in layer_shapes]),max([ls[1] for ls in layer_shapes]))
    
    layers = [im_resize(layer,ls0) for layer in layers]
    layer_shapes = [layer.shape for layer in layers]
    print('new layer shapes',layer_shapes)
    return np.column_stack([layer.reshape((ls[0]*ls[1],ls[2])) for layer,ls in zip(layers,layer_shapes)]),ls0

def im_resize(layer,sh):
    if layer.ndim > 2:
        new_sh = sh + (layer.shape[2],)
    else:
        new_sh = sh + (1,)
    x = np.resize(layer,new_sh)
    return x
    

                
             
             
    
                
