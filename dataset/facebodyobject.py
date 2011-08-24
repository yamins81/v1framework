from data_cache import cache_join, cache_open
import dataset
import column
import sys
import os
import boto

def fbo_join(*names):
    return cache_join('facebodyobject', *names)

class FaceBodyObject_blob(dataset.DatasetBlob):
    global_instance = None
    @classmethod
    def get_global_instance(cls):
        if cls.global_instance is None:
            cls.global_instance = cls()
        return cls.global_instance
        
    def fetch(self):
        dataset_dir = fbo_join()
        zip_path = fbo_join('FaceBodyObject_2011_08_03.tar.gz')
        conn = boto.connect_s3()
        b = conn.get_bucket('dicarlocox-datasets')
        k = b.get_key('FaceBodyObject_2011_08_03.tar.gz')
        k.get_contents_to_filename(zip_path)
        os.system('cd ' + dataset_dir + '; tar -xzvf FaceBodyObject_2011_08_03.tar.gz')
        
        
    def load(self):
        dataset_dir = fbo_join()
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset_path = fbo_join('FaceBodyObject_2011_08_03') 
        if not os.path.exists(dataset_path):
            self.fetch()
        files = os.listdir(dataset_path)
        metadata = []
        for f in files:
            if f.endswith('.png'):
                num = int(f.split('.')[0][2:])
                if num <= 20:
                    subject = 'face'
                elif num <= 40:
                    subject = 'body'
                else:
                    subject = 'object'
    
                d = {'jgpfile':f,
                 'subject': subject} 
        
                metadata.append(d)
        
        return metadata

class ImgFullPath(column.MapColumn):
    @staticmethod
    def _fn(dct):
        return fbo_join(dct['name'], dct['jpgfile'])

class FaceBodyObject(dataset.Dataset):
    def __init__(self):
        self.blob = FaceBodyObject_blob.get_global_instance()
        thing = self.blob.load()
        columns = {}
        columns['meta'] = column.ListColumn(thing)
        columns['img_fullpath'] = ImgFullPath(columns['meta'])

        dataset.Dataset.__init__(self, columns)

def facebodyobject_from_son(doc):
    return FaceBodyObject()


