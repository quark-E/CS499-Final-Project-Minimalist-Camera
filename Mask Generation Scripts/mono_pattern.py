import numpy as np
from PIL import Image, ImageDraw
import random
import sys

WIDTH = 800
HEIGHT = 480
NUM_CELLS = 60
BLOCK_RATIO = 0.7
ANCHOR_BORDER = True   

def generate_mono_mask(filename):
    print(f"Generating Grayscale mask...")

    w_proc = WIDTH // 4
    h_proc = HEIGHT // 4
    
    seeds_x = np.random.randint(0, w_proc, NUM_CELLS)
    seeds_y = np.random.randint(0, h_proc, NUM_CELLS)
    
    seed_colors = []
    for _ in range(NUM_CELLS):
        if random.random() < BLOCK_RATIO:
            seed_colors.append((0, 0, 0))
        else:
            seed_colors.append((255, 255, 255))

    y, x = np.indices((h_proc, w_proc))
    output_buffer = np.zeros((h_proc, w_proc, 3), dtype=np.uint8)
    min_dists = np.full((h_proc, w_proc), np.inf)

    for i in range(NUM_CELLS):
        sx, sy = seeds_x[i], seeds_y[i]
        color = seed_colors[i]
        dist = (x - sx)**2 + (y - sy)**2
        mask = dist < min_dists
        min_dists[mask] = dist[mask]
        output_buffer[mask] = color

    img = Image.fromarray(output_buffer, 'RGB')
    img = img.resize((WIDTH, HEIGHT), Image.NEAREST)

    if ANCHOR_BORDER:
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, WIDTH-1, HEIGHT-1], outline=(255, 255, 255), width=1)

    print(f"Saving to {filename}")
    img.save(filename)

if __name__ == "__main__":
    generate_mono_mask("mono_pattern.png")
