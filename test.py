from cv_utils import camera
from cv_utils import file_utils as fu
from cv_utils import geometry_utils as geo
from cv_utils import image_utils as iu

cam_dict = fu.read_yaml("./images/dscam_examples/info.yaml")

ds_image = fu.read_image("./images/dscam_examples/image.png")

dscam = camera.DoubleSphereCamera(cam_dict)

# pinhole = camera.PinholeCamera.from_fov([320,320],60.)
equirect = camera.EquirectangularCamera(
    {   
        "image_size": (512,256),
        "min_phi_deg": -90.,
        "max_phi_deg": 90.
    })

# warp_image = geo.transition_camera_view(ds_image,dscam,pinhole)
warp_image = geo.transition_camera_view(ds_image,dscam,equirect)

iu.show_two_images(ds_image,warp_image)