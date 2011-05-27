import sys
import cPickle
import hashlib
import os
import random
import multiprocessing
import functools 

from sge_utils import qsub


def generate_single_image(config,returnfh = False):
	orig_dir = os.getcwd()
	os.chdir(os.path.join(os.environ['HOME'] , 'render_wd'))
	tmp = tempfile.mkdtemp()
	renderer.render(tmp,[config])
	imagefile = [os.path.join(tmp,x) for x in os.listdir(tmp) if x.endswith('.tif')][0]
	os.chdir(orig_dir)
	
	image_string = open(imagefile).read()
		
	outfile = os.path.join(outdir,x['image']['id'] + '.tif')
    F = open(outfile,'w')
    F.write(image_string)
    F.close()


import json

def generate_images_parallel(outdir,id_list):

    jobids = []
    for (i,id) in enumerate(id_list):
        url = 'http://50.19.109.25:9999/3dmodels?query={"id":"' + id + '"}'
        y = json.loads(urllib.urlopen(url).read())
        x = {'bg_id':'INTERIOR_10SN.tdl',
             'model_params':[{'model_id':y['id'],
                              'rxy':y.get('canonical_view',{}).get('rxy',0),
                              'rxz':y.get('canonical_view',{}).get('rxz',0),
                              'ryz':y.get('canonical_view',{}).get('ryz',0)
                             }]
            }
        jobid = qsub(generate_single_image,(x,outdir),opstring='-pe orte 2 -l qname=rendering.q -o /home/render -e /home/render')  
        jobids.append(jobid)
        
    
    return {'child_jobs':jobids}
    
 
 
 