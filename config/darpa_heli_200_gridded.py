from bson import SON

images = SON([('selection','gridded'),
                 ('generator','darpa'),
                 ('sizes',[(200,200)]),
                 ('offsets',[(0,0)]),
                 ('base_dir','../../darpa/helidata')])

config = {'images' : [images]}
