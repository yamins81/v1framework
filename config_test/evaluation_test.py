from bson import SON

config = {
      
     'train_test' : [    

		   SON([
		      ('transform_average', SON([('transform_name','translation_and_orientation')])),
			  ('N',5), 
			  ('ntrain',10),
			  ('ntest',10),
			  ('ntrain_pos',5),
			  ('ntest_pos',5),
			  ('query',SON([('image.object.type','rectangle')]))
			  ])		  
     ]       
     
}