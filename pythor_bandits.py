"""
"""
import numpy
from hyperopt.genson_bandits import GensonBandit
               
        
def fill_in_default(a,b):
    if hasattr(a,'keys'):
        assert hasattr(b,'keys'):
        for k in b.keys():
            if k not in a:
                a[k] = b[k]
            else:
                fill_in_default(a[k],b[k])
    elif hasattr(a,'iter'):
        assert len(a) == len(b)
        for (aa,bb) in zip(a,b):
            fill_in_default(aa,bb)

class EvaluationBandit(GensonBandit):

    def __init__(self, source_string,
                 image_hash,
                 model_hash,
                 image_config_gen,
                 opt_hash,
                 convolve_func_name,
                 use_db = False):
    
        super(EvaluationBandit,self).__init__(source_string=source_string)
        
        connection = pm.Connection(document_class = SON)
    	db = connection['thor']
	    perf_col = db['performance']
	    split_fs = gridfs.GridFS(db, 'splits')
	    splitperf_fs = gridfs.GridFS(db, 'split_performance')

		self.image_hash = image_hash
		self.image_config_gen = image_config_gen
		self.opt_hash = opt_hash
		self.convolve_func_name = convolve_func_name
		self.use_db = use_db
		self.model_hash = model_hash
	
    def evaluate(self,argd,ctrl):
        model,task = self.model_and_task_from_template(argd)
        performance = self.get_performance(model,task)
        return dict(loss = performance, status = 'ok')

    def model_task_from_template(self,template):
        model = template['model']
		default = template['default_model']
		fill_in_default(model, default)
		task = template['task']
		return model, task
		
    def get_performance(self,model,task):
	
		perf_col = self.perf_col
		split_fs = self.split_fs
		opt_hash = self.opt_hash
		convolve_func_name = self.convolve_func_name
		image_hash = self.image_hash
		image_config_gen = self.image_config_gen
		use_db = self.use_db
		
		splits = generate_splits(task,image_hash,'images',
		                         overlap=task.get('overlap'),
		                         balance=task.get('balance')) 
		for (ind,split) in enumerate(splits):
			put_in_split(split,image_config_gen,model,task,
			             opt_hash,ind,split_fs)
			print('evaluating split %d' % ind)
			res = extract_and_evaluate_core(split,model,convolve_func_name,
			                                task,use_db=use_db)    
			put_in_split_result(res,image_config_gen,model,task,
			                    opt_hash,ind,splitperf_fs)
			split_results.append(res)
			perf = put_in_performance(split_results,image_config_gen,model,
			                          model_hash,image_hash,
			                          perf_col,task,opt_hash)
		
		return perf