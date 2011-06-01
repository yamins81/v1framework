from starflow.protocols import Apply, Applies

import deploy

@Applies(deploy.images,args=('../config/ten_categories_images.py',True))
def ten_categories_images():
    Apply()

@Applies(deploy.images,args=('../config/polygon_task.py',False))
def polygon_task_images():
    Apply()