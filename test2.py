from module.file_utils import read_image
from module.plot import draw_line_by_points, show_image


image = read_image("/home/sehyun/workspace/computer_vision_python/replica/scan1/000000_rgb.png")


pt1 = (246,110)
pt2 = (134,222)
image = draw_line_by_points(image,pt1,pt2)
show_image(image)
