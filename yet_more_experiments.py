from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols


@Applies(deploy.images,args=('../config/faces_images.py',True))
def generate_faces_images():
    Apply()

@protocolize()
def ext_eval_various_random_l3_face_subtasks(depends_on=('../config/face_subtasks.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/faces_images.py')):
    """
    """
    protocols.extract_and_evaluate_protocol('../config/face_subtasks.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/faces_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

