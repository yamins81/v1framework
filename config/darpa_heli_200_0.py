from bson import SON

images = SON([('selection','random'),
                 ('generator','darpa'),
                 ('size',(200,200)),
                 ('num_images',100000),
                 ('base_dir','../../darpa/helidata'), 
                 ('enrich_positives',True)])

config = {'images' : [images]}
