from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols



##############################################
###############varied rotations###############

@Applies(deploy.images,args=('../config/dicarlo_movies.py',True))
def generate_rotated_images():

    Apply()
    
    
