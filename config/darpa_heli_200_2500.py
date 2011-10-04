from bson import SON

images = SON([('selection','random'),
                 ('generator','darpa'),
                 ('size',(200,200)),
                 ('num_images',2500),
                 ('base_dir','../../darpa/helidata')])

config = {'images' : [images]}
