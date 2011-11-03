from bson import SON

images = SON([('selection','specific'),
                 ('generator','darpa'),
                 ('image_extension','.png'),
                 ('size',(200,200)),
                 ('annotate_dir','../../darpa/nv2_detections/'),
                 ('base_dir','../../darpa/helidata_test_all'),
                 ('groundtruth_dir','../../darpa/Heli_iRobot_annotated/'),
                 ('frame_correspondence','../../darpa/test_frame_correspondence.csv')])

config = {'images' : [images]}
