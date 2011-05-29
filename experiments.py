from starflow.protocols import Apply, Applies

import deploy

@Applies(deploy.images,args=('../config/ten_categories_images.py',True))
def ten_categories_images():
    Apply()
