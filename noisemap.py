import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from perlin_noise import PerlinNoise

GRID_SIZE = 90
OCTAVES = 1
SPEED = 0.05
TRANSITION_SPEED = 0.02
current_scale = 0.06

noise = PerlinNoise(octaves=OCTAVES)
vectorized_noise = np.vectorize(lambda x, y, z: noise([x, y, z]))

bg_current = np.random.rand(3)
bg_target = np.random.rand(3)

net_current = np.random.rand(3)
net_target = np.random.rand(3)

plt.rcParams['toolbar'] = 'None'
fig, ax = plt.subplots(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

X, Y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
rgb_data = np.zeros((GRID_SIZE, GRID_SIZE, 3))

im = ax.imshow(rgb_data, origin='lower', interpolation='bicubic')
ax.axis('off')

def on_scroll(event):
    global current_scale
    if event.button == 'up':
        current_scale *= 0.85
    elif event.button == 'down':
        current_scale *= 1.15
    current_scale = max(0.005, min(0.3, current_scale))

fig.canvas.mpl_connect('scroll_event', on_scroll)

def update(frame):
    global bg_current, bg_target, net_current, net_target
    
    bg_current += (bg_target - bg_current) * TRANSITION_SPEED
    if np.linalg.norm(bg_target - bg_current) < 0.05:
        bg_target = np.random.rand(3)
        
    net_current += (net_target - net_current) * TRANSITION_SPEED
    if np.linalg.norm(net_target - net_current) < 0.05:
        net_target = np.random.rand(3)
    
    z_coord = frame * SPEED
    raw_noise = vectorized_noise(X * current_scale, Y * current_scale, z_coord)
    
    net_intensity = (1.0 - np.abs(raw_noise)) ** 5.0
    net_intensity = np.clip(net_intensity, 0.0, 1.0)
    
    t = net_intensity[:, :, np.newaxis]
    rgb_image = (1.0 - t) * bg_current + t * net_current
    rgb_image = np.clip(rgb_image, 0.0, 1.0)

    im.set_array(rgb_image)
    return [im]

ani = FuncAnimation(fig, update, interval=16, blit=True)

plt.show()
