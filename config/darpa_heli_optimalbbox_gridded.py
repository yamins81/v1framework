from bson import SON

images = SON([('selection','gridded'),
                 ('generator','darpa'),
                 ('sizes',[(66,136)]),
                 ('offsets',[(0,0),(33,68)]),
                 ('base_dir','../../darpa/helidata')])

config = {'images' : [images]}
