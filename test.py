from cv_utils import camera


cam = camera.PinholeCamera.from_fov((1920,1080),60.)

print(cam.K)
