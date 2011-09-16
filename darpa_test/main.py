import scipy as sp
import Image
import ImageOps
import tabular as tb
import pythor3.wildwest.bbox as bbox

import filter_generation as fg
from processing import image2array, preprocess, postprocess

from pythor_protocols import fbcorr

from scikits.learn import linear_model

def get_image(file):
    im = ImageOps.grayscale(Image.open(file))
    return sp.misc.fromimage(im)
    
def detect_and_evaluate(model,metadatafile,imagedir,empties_dir,train_objects,test_frames,num_empties=0):

    filters = fg.get_hierarchical_filterbanks(model['config']['model']['layers'])
    
    metadata = tb.tabarray(SVfile=metadatafile)
    #get labels for training objects
    train_objects = tb.tabarray(records = train_objects,names=['frame_number','object_number'])
    train_labels_inds = []
    label_fields = ['BoundingBox_X1', 
                    'BoundingBox_Y1', 
                    'BoundingBox_X2', 
                    'BoundingBox_Y2', 
                    'BoundingBox_X3', 
                    'BoundingBox_Y3', 
                    'BoundingBox_X4', 
                    'BoundingBox_Y4', 
                    'ObjectType']
    
    metadata_1 = metadata[label_fields]
    for t in train_objects:
        inds = (metadata['frame_number'] == t['frame_number']).nonzero()[0]
        ind = inds[t['object_number']]
        train_label_inds.append(ind)
    train_labels = metadata[train_labels_inds]
    #get stamps for training objects
    xfields = ['BoundingBox_X1', 'BoundingBox_X2',  'BoundingBox_X3','BoundingBox_X4']
    yfields = ['BoundingBox_Y1', 'BoundingBox_Y2',  'BoundingBox_Y3','BoundingBox_Y4']

    train_points = []
    train_points_labels = []
    sizes = []
    for obj,label in zip(train_objects,train_labels):
        framenumber = str(obj['frame_number'])
        framenumber = '0'*(5 - len(framenumber)) + framenumber
        framefile = os.path.join(imagedir,framenumber + '.jpg')
        im = get_image(framefile)
        box = bbox.BoundingBox(xs = [obj[xf] for xf in xfields],
                               ys = [obj[yf] for yf in yfields])
        stamp = bbox.stamp(im,box)
        sizes.append((box.width,box.height))
        features = get_features(model,filters,stamp)
        feature_points = get_feature_points(features)
        train_points.extend(feature_points)
        
        vecs = get_positions((box.width,box.height),tf)
        train_points_labels.extend(vecs)

    empty_files = [os.path.join(empties_dir,x) for x in os.listdir(empties_dir) if x.endswith('.jpg')]
    for ind in range(num_empties):
        sz = sizes[sp.random.randint(0,high=len(sizes))]
        im,box = get_random_empty_bbox(empty_files,sz)
        stamp = bbox.stamp(im,box)
        features = get_features(model,filters,stamp)
        feature_points = get_feature_points(features)
        train_points.extend(feature_points)
        
        vecs = [(-100,-100) for ind in range(len(feature_points))]
        train_points_labels.extend(vecs)

    train_points = np.array(train_points)
    train_points_labels = np.array(train_points_labels)
    
    #run regression
    clf = linear_model.LinearRegression()
    clf.fit(train_points,train_points_labels)    
    
    #extract features from test frames
    predictions = []
    for framefile in test_frames:
        im = get_image(framefile)
        tf = get_features(model,filters,im)
        test_points = get_feature_points(tf)
        predictions.append(clf.predict(test_points))

    return clf,predictions
    
def get_random_empty_bbox(empty_files,shp):
    fl = empty_files[np.random.randint(len(empty_files))]    
    im = get_image(fl)
    sx,sy = im.size
    start_x = np.random.randit(sx - shp[0])
    start_y = np.random.randit(sy - shp[1])
    
    return im[start_x : start_x + shp[0] , start_y : start_y + shp[1]]
    
def get_features(model_config, filters, array):
 
    m_config = model_config['config']['model']
    
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
        assert len(filters) == len(layers)
        dtype = array[0].dtype
        
        array_dict = {}
        for (ind,(filter,layer)) in enumerate(zip(filters,layers)):
            if feed_up:
                array_dict[ind-1] = array
        
            if filter is not None:
                array = fbcorr(array, filter, layer , convolve_func)
            
            if layer.get('lpool'):
                array = lpool(array,conv_mode,layer['lpool'])

            if layer.get('lnorm'):
                if layer['lnorm'].get('use_old',False):
                    array = old_norm(array,conv_mode,layer['lnorm'])
                else:
                    array = lnorm(array,conv_mode,layer['lnorm'])

        array_dict[len(layers)-1] = array
            
        return array_dict

    

def get_positions(s,tf):
    layers = tf.values()
    layer_shapes = [layer.shape for layer in layers]
    ls0 = layer_shapes[0]
    assert all([ls[:2] == ls0[:2] for ls in layer_sizes])    
    
    xpos = (np.arange(ls0[0])/ls[0])*s[0]
    ypos = (np.arange(ls0[1])/ls[1])*s[1]
    
    return [(x,y) for y in ypos for x in xpos]
    

def get_feature_points(tf):
    layers = tf.values()
    layer_shapes = [layer.shape for layer in layers]
    ls0 = layer_shapes[0]
    assert all([ls[:2] == ls0[:2] for ls in layer_sizes])
    return np.column_stack([layer.reshape((ls[0]*ls[1],ls[2])) for layer,ls in zip(layers,layer_shapes)])
    

                
             
             
    
                