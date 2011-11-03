from bson import SON

images = SON([('selection','gridded'),
                 ('generator','darpa'),
                 ('image_extension','.png'),
                 ('sizes',[(66,136)]),
                 ('offsets',[(0,0),(33,68)]),
                 ('base_dir','../../darpa/helidata_test')])

config = {'images' : [images]}
