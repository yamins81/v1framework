from bson import SON

images = SON([('selection','gridded'),
                 ('generator','darpa'),
                 ('image_extension','.png'),
                 ('sizes',[(200,200)]),
                 ('offsets',[(0,0),(50,50),(100,100),(125,125)]),
                 ('base_dir','../../darpa/helidata_test_all')])

config = {'images' : [images]}
