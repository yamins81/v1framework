import scipy as sp
import numpy as np
from scikits.learn import svm
from scikits.learn.linear_model.logistic import LogisticRegression

'''
SVM classifier module
'''
    

def classify(train_features,
             train_labels,
             test_features,
             test_labels, 
             classifier_kwargs
            ):

    '''Classify data and return
        accuracy
        area under curve
        average precision
        and svm raw data in a dictianary'''

    #mapping labels to 0,1
    labels = sp.unique(sp.concatenate((train_labels, test_labels)))
    assert labels.size == 2
    label_to_id = dict([(k,v) for v, k in enumerate(labels)])

    train_ys = sp.array([label_to_id[i] for i in train_labels])
    test_ys = sp.array([label_to_id[i] for i in test_labels])

    #train
    model,fmean,fstd = classifier_train(train_features, train_ys, test_features,**classifier_kwargs)

    #test
    if classifier_kwargs.get('classifier_type') == 'MCC':
        weights = model.coef_
        bias = model.intercept_
    else:
        weights = model.coef_.ravel()
        bias = model.intercept_.ravel()
    
    test_predictor = sp.dot(test_features, weights) + bias    
    test_prediction = model.predict(test_features)
    train_prediction = model.predict(train_features)

    #raw data to be saved for future use
    cls_data = {'test_prediction' : test_prediction,  
                'test_labels' : test_labels, 
                'coef' : model.coef_, 
                'intercept' : model.intercept_,
                'train_mean' : fmean,
                'train_std': fstd,
                'test_margins' : test_predictor,
               }

    #accuracy
    test_accuracy = 100*(test_prediction == test_ys).sum()/float(len(test_ys))
    train_accuracy = 100*(train_prediction == train_ys).sum()/float(len(train_ys))
    
    #precison and recall
    c = test_predictor
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(test_ys[si] == 1))
    fp = sp.cumsum(sp.single(test_ys[si] == 0))
    rec = tp /sp.sum(test_ys > 0)
    prec = tp / (fp + tp)
    
    ap = 0
    rng = sp.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size

    #area under curve
    h = sp.diff(rec)
    auc = sp.sum(h * (prec[1:] + prec[:-1])) / 2.0


    return {'auc':auc,
            'ap':ap, 
            'train_accuracy': train_accuracy,
            'test_accuracy' : test_accuracy,
            'cls_data':cls_data
           }


        
def ova_classify(train_features,
                     train_labels,
                     test_features,
                     test_labels,
                     classifier_kwargs):
                     
    """
    Classifier using one-vs-all on top of liblinear binary classification.  
    Computes mean average precision (mAP) and mean area-under-the-curve (mAUC)
    by averaging these measure of the binary results. 
    """
                     
    train_features, test_features,fmean,fstd = __sphere(train_features, test_features)

    classifier_kwargs['sphere'] = False
    
    labels = sp.unique(sp.concatenate((train_labels, test_labels)))
    label_to_id = dict([(k,v) for v, k in enumerate(labels)])

    train_ids = sp.array([label_to_id[i] for i in train_labels])
    test_ids = sp.array([label_to_id[i] for i in test_labels])
    all_ids = sp.array(range(len(labels)))

    classifiers = []
    aps = []
    aucs = []
    cls_datas = []
    test_accuracies = []
    train_accuracies = []

    signs = []
    for id in all_ids: 
        binary_train_ids = sp.array([2*int(l == id) - 1 for l in train_ids])
        binary_test_ids = sp.array([2*int(l == id) - 1 for l in test_ids])
        signs.append(binary_train_ids[0])   
        
        res = classify(train_features, binary_train_ids, test_features, binary_test_ids,classifier_kwargs)
        
        
        aps.append(res['ap'])
        aucs.append(res['auc'])
        test_accuracies.append(res['test_accuracy'])
        train_accuracies.append(res['train_accuracy'])
        cls_datas.append(res['cls_data'])
    
    mean_ap = sp.array(aps).mean()
    mean_auc = sp.array(aucs).mean()
    
    signs = sp.array(signs)
    weights = signs * (sp.row_stack([cls_data['coef'] for cls_data in cls_datas]).T)
    bias = signs * (sp.row_stack([cls_data['intercept'] for cls_data in cls_datas]).T)
    
    predictor = max_predictor(weights,bias,labels)
  
    test_prediction = predictor(test_features)
    test_accuracy = float(100*(test_prediction == test_labels).sum() / float(len(test_prediction)))

    train_prediction = predictor(train_features)
    train_accuracy = float(100*(train_prediction == train_labels).sum() / float(len(train_prediction)))

    cls_data = {'coef' : weights, 
     'intercept' : bias, 
     'train_labels': train_labels,
     'test_labels' : test_labels,
     'train_prediction': train_prediction, 
     'test_prediction' : test_prediction,
     'labels' : labels
     }


    return {'cls_data' : cls_data,
     'train_accuracy' : train_accuracy,
     'test_accuracy' : test_accuracy,
     'mean_ap' : mean_ap,
     'mean_auc' : mean_auc,
     'test_accuracies' : test_accuracies,
     'train_accuracies' : train_accuracies
     }
     

def multi_classify(train_features,
                     train_labels,
                     test_features,
                     test_labels,
                     multi_class = False):
    """
    Classifier using the built-in multi-class classification capabilities of liblinear
    """

    labels = sp.unique(sp.concatenate((train_labels, test_labels)))
    label_to_id = dict([(k,v) for v, k in enumerate(labels)])
 
    train_ids = sp.array([label_to_id[i] for i in train_labels])
    test_ids = sp.array([label_to_id[i] for i in test_labels])
    
    classifier = classifier_train(train_features, train_ids, test_features, multi_class = multi_class)[0]
    weights = classifier.coef_.T
    bias = classifier.intercept_
        
    test_prediction = labels[classifier.predict(test_features)]
    test_accuracy = float(100*(test_prediction == test_labels).sum() / float(len(test_prediction)))
    train_prediction = labels[classifier.predict(train_features)]
    train_accuracy = float(100*(train_prediction == train_labels).sum() / float( len(train_prediction)))
    
    margin_fn = lambda v : (sp.dot(v,weights) + bias)
    test_margins = margin_fn(test_features)
#    test_margin_prediction = labels[test_margins.argmax(1)]
    train_margins = margin_fn(train_features)
#    train_margin_prediction = labels[train_margins.argmax(1)]
#    assert (test_prediction == test_margin_prediction).all(), 'test margin prediction not correct'
#    assert (train_prediction == train_margin_prediction).all(), 'train margin prediction not correct'    
    
    mean_ap,mean_auc = multiclass_stats(test_labels,test_prediction,labels)

    cls_data = {'coef' : weights, 
     'intercept' : bias, 
     'train_labels': train_labels,
     'test_labels' : test_labels,
     'train_prediction': train_prediction, 
     'test_prediction' : test_prediction,
     'labels' : labels,
     'test_margins' : test_margins,
#     'train_margins' : train_margins
     }


    return {'cls_data' : cls_data,
     'train_accuracy' : train_accuracy,
     'test_accuracy' : test_accuracy,
     'mean_ap' : mean_ap,
     'mean_auc' : mean_auc
     }
 

def multiclass_stats(actual,predicted,labels):
    aps = []
    aucs = []
    
    for label in labels:
        prec,rec = precision_and_recall(actual,predicted,label)
        ap = ap_from_prec_and_rec(prec,rec)
        aps.append(ap)
        auc = auc_from_prec_and_rec(prec,rec)
        aucs.append(auc)
    
    mean_ap = np.array(aps).mean()
    mean_auc = np.array(aucs).mean()
    
    return mean_ap,mean_auc


def precision_and_recall(actual,predicted,cls):
    c = (actual == cls)
    si = sp.argsort(-c)
    tp = sp.cumsum(sp.single(predicted[si] == cls))
    fp = sp.cumsum(sp.single(predicted[si] != cls))
    rec = tp /sp.sum(predicted == cls)
    prec = tp / (fp + tp)
    return prec,rec

    
def ap_from_prec_and_rec(prec,rec):
    ap = 0
    rng = sp.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def auc_from_prec_and_rec(prec,rec):
    #area under curve
    h = sp.diff(rec)
    auc = sp.sum(h * (prec[1:] + prec[:-1])) / 2.0
    return auc        
    
    
def classifier_train(train_features,
                     train_labels,
                     test_features,
                     classifier_type = "liblinear",
                     sphere = True,
                     **kwargs
                     ):
    """ Classifier training using SVMs

    Input:
    train_features = training features (both positive and negative)
    train_labels = corresponding label vector
    svm_eps = eps of svm
    svm_C = C parameter of svm
    classifier_type = liblinear or libsvm"""
       
    #sphering
    if sphere:
        train_features, test_features,fmean,fstd = __sphere(train_features, test_features)
    else:
        fmean = None
        fstd = None

    if classifier_type == 'liblinear':
        clf = svm.LinearSVC(**kwargs)
    if classifier_type == 'libSVM':
        clf = svm.SVC(**kwargs)
    elif classifier_type == 'LRL':
        clf = LogisticRegression(**kwargs)
    elif classifier_type == 'MCC':
        clf = CorrelationClassifier(**kwargs)

    clf.fit(train_features, train_labels)
    
    return clf,fmean, fstd

#sphere data
def __sphere(train_data, test_data):
    '''make data zero mean and unit variance'''

    fmean = train_data.mean(0)
    fstd = train_data.std(0)

    train_data -= fmean
    test_data -= fmean
    fstd[fstd==0] = 1
    train_data /= fstd
    test_data /= fstd

    return train_data, test_data, fmean, fstd
     
def max_predictor(weights,bias,labels):
    return lambda v : labels[(sp.dot(v,weights) + bias).argmax(1)]

def liblinear_predictor(clas, bias, labels):
    return lambda x : labels[liblinear_prediction_prediction_function(x,clas,labels)]

def liblinear_prediction_function(farray , clas, labels):

    if len(labels) > 2:
        nf = farray.shape[0]
        nlabels = len(labels)
        
        weights = clas.raw_coef_.ravel()
        nw = len(weights)
        nv = nw / nlabels
        
        D = np.column_stack([farray,np.array([.5]).repeat(nf)]).ravel().repeat(nlabels)
        W = np.tile(weights,nf)
        H = W * D
        H1 = H.reshape((len(H)/nw,nv,nlabels))
        H2 = H1.sum(1)
        predict = H2.argmax(1)
        
        return predict
    else:
    
        weights = clas.coef_.T
        bias = clas.intercept_
        
        return (1 - np.sign(np.dot(farray,weights) + bias) )/2
        

#=-=-=-=-=-=-=-=
#maximum correlation
#=-=-=-=-=-=-=-=

def uniqify(seq, idfun=None): 
    '''
    Relatively fast pure python uniqification function that preservs ordering
    ARGUMENTS:
        seq = sequence object to uniqify
        idfun = optional collapse function to identify items as the same
    RETURNS:
        python list with first occurence of each item in seq, in order
    '''
    try:

        # order preserving
        if idfun is None:
            def idfun(x): return x
        seen = {}
        result = []
        for item in seq:
            marker = idfun(item)
            # in old Python versions:
            # if seen.has_key(marker)
            # but in new ones:
            if marker in seen: continue
            seen[marker] = 1
            result.append(item)
    except TypeError:
        return [x for (i,x) in enumerate(seq) if x not in seq[:i]]
    else:
        return result


class CorrelationClassifier():

    def __init__(self):
        pass
        
    def fit(self,train_features,train_labels):
        self.labels = uniqify(train_labels)
        self.coef_ = np.array([train_features[train_labels == label].mean(0) for label in self.labels]).T
        self.intercept_ = -.5*(self.coef_ ** 2).sum(0)
        self.nums = [len((train_labels == label).nonzero()[0]) for label in self.labels]
             
    def predict(self,test_features):
        prediction = self.prediction_function(test_features)
        return [self.labels[i] for i in prediction]
          
    def prediction_function(self,test_features):
        return self.decision_function(test_features).argmax(1)
        
    def decision_function(self,test_features):
        return np.dot(test_features,self.coef_) + self.intercept_
        
    def update_fit(self,new_features,new_labels):
        unique_new_labels = uniqify(new_labels)
        for new_label in unique_new_labels: 
            new_f = new_features[new_labels == new_label]
            new_num = new_f.shape[0]
            if new_label in self.labels:
                l_ind = self.labels.index(new_label)
                num = self.nums[l_ind]
                self.coef_[:,l_ind] = (num * self.coef_[:,l_ind] + new_num * new_f.mean()) / (num + new_num)
                self.intercept_[l_ind] = -.5 * (self.coef_[:,l_ind] ** 2).sum()
                self.nums[l_ind] += new_num
            else:
                new_coef = np.empty((self.coef_.shape[0],self.coef_.shape[1] + 1))
                new_intercept = np.empty((self.intercept_.shape[0] + 1,))
                new_coef[:,:-1] = self.coef_
                new_intercept[:-1] = self.intercept_
                
                new_coef[:,-1] = new_f.mean()
                new_intercept[-1] = -.5 * (new_coef[:,-1] **2).sum()
                
                self.coef_ = new_coef
                self.intercept_ = new_intercept
                self.labels.append(new_label)
                self.nums.append(new_num) 
