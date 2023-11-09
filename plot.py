import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data, plot_2d_box, plot_3d_box, plot_gcp, plot_psi, plot_hull, plot_min_area_rect

ts = 56
image_pv, image_tv, image_seg, vehicles_pv, vehicles_tv = get_data(ts, base_url='/Users/tobias/ziegleto/data/5Safe/carla/circle/', w=1920, h=1080)

_ = plt.imshow(image_pv)
for vehicle in vehicles_pv:
    #plot_3d_box(np.array(vehicle['bb']))
    #plot_2d_box(np.array(vehicle['bb']))
    plot_gcp(vehicle['gcp'], size=100)
    #plot_psi(vehicle['gcp'], vehicle['psi'])
    #plot_hull(vehicle['hull'])
    #plot_min_area_rect(vehicle['min_area_rect'])

plt.show()