import os
import time
import random

import pymongo
from bson import SON

import pythor3
import column

def get_random_name():
     return ''.join(map(str,[random.randint(0,9) for i in range(30)]))

class MongoColumn(column.Column):
 
    META_DB_NAME = 'metadata'
    META_DB_LOCATION = os.path.join(pythor3.get_home(),'mongo')
    DB_CONFIG_PATH = os.path.join(META_DB_LOCATION,'conf')
    META_DB_PORT = 45678
    DB_CONFIG_TEXT = """
dbpath = %s
bind_ip = 127.0.0.1
port = %d
logpath = %s""" % (os.path.join(META_DB_LOCATION,'db'),
                   META_DB_PORT,
                   os.path.join(META_DB_LOCATION,'log'))

    def initialize_db(self):
        if not os.path.exists(self.META_DB_LOCATION):
            os.makedirs(self.META_DB_LOCATION)
            with open(self.DB_CONFIG_PATH,'w') as f:
                f.write(self.DB_CONFIG_TEXT)  
                
        if not os.path.exists(os.path.join(self.META_DB_LOCATION,'db')):
            os.makedirs(os.path.join(self.META_DB_LOCATION,'db'))

        try:
            self._conn = pymongo.Connection(port=self.META_DB_PORT,document_class=SON)
        except pymongo.errors.ConnectionFailure:
            print('mongodb connection failed, trying to start mongod process')
            os.system('mongod run --config ' + self.DB_CONFIG_PATH + ' --fork')
            time.sleep(10)
            self._conn = pymongo.Connection(port=self.META_DB_PORT,document_class=SON)

        self._meta_coll_name = get_random_name()
        self._db = self._conn[self.META_DB_NAME]
        self._db.drop_collection(self._meta_coll_name)
        self._meta_coll = self._db[self._meta_coll_name]
        for (ind,m) in enumerate(self._col):
            assert '_id' not in self._col, 'meta object elements cannot have field named "_id"'
            m['_id'] = ind
            self._meta_coll.insert(m,safe=True)
            print ind
                
    def __init__(self, col):
        self._col = col
        self.initialize_db()
        column.Column.__init__(self, len(col))

    def __del__(self):
        try:
            db = self._db
        except AttributeError:
            return
        db.drop_collection(self._meta_coll_name)
        
    def __getitem__(self, idx):
        return self._col[idx]

    def query(self, spec):
        """
        Return (data, meta) pairs for which `meta` matches specification
        `spec`.

        See Mongodb docs for documentation of query specification.
        http://www.mongodb.org/display/DOCS/Advanced+Queries
        """
        idx = self.query_idx(spec)
        return self[idx]

    def query_idx(self, spec):
        """Return an idx array or iterator for where spec matches meta.

        See Mongodb docs for documentation of query specification.
        http://www.mongodb.org/display/DOCS/Advanced+Queries
        """
        return [x['_id'] for x in self._meta_coll.find(spec,fields=['_id'])]

