from starflow.protocols import Apply, Applies

from deploy import images

@Applies(deploy.images,args=('../config/ten_categories_images.py',True))
def ten_categories_images():
    Apply()
