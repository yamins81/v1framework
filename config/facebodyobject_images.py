from bson import SON

facebodyobject = SON([('selection','dataset_api'),
                 ('generator','dataset_api'),
                 ('run_with','dataset.facebodyobject.facebodyobject_from_son'),
                 ('dataset_name','facebodyobject')])

config = {'images' : [facebodyobject]}
