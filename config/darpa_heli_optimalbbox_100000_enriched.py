from bson import SON

images = SON([('selection','random'),
                 ('generator','darpa'),
                 ('size',(66,136)),
                 ('num_images',50000),
                 ('enrich_positives',True),
                 ('base_dir','../../darpa/helidata')])

config = {'images' : [images]}
