import numpy as np
from PIL import Image, ImageDraw
import random
import sys

WIDTH = 800
HEIGHT = 480
NUM_CELLS = 60         
BLOCK_RATIO = 0.20      
DARK_MODE = True
ANCHOR_BORDER = True

DIM_RANGE = (80, 150)
BRIGHT_RANGE = (180, 255)
HIGHLIGHT_CHANCE = 0.3

def generate_worley_mask(filename):
    print(f"Generating {NUM_CELLS} Voronoi cells...")
    
    w_proc = WIDTH // 4
    h_proc = HEIGHT // 4
    
    seeds_x = np.random.randint(0, w_proc, NUM_CELLS)
    seeds_y = np.random.randint(0, h_proc, NUM_CELLS)
    
    seed_colors = []
    for _ in range(NUM_CELLS):
        if random.random() < BLOCK_RATIO:
            if DARK_MODE:
                r = random.choice([0, 5, 10])
                g = random.choice([0, 5, 10])
                b = random.choice([0, 5, 10])
                seed_colors.append((r, g, b))
            else:
                seed_colors.append((0, 0, 0))
        else:
            if random.random() < HIGHLIGHT_CHANCE:
                low, high = BRIGHT_RANGE
            else:
                low, high = DIM_RANGE
            
            seed_colors.append((
                random.randint(low, high),
                random.randint(low, high),
                random.randint(low, high)
            ))

    y, x = np.indices((h_proc, w_proc))
    
    output_buffer = np.zeros((h_proc, w_proc, 3), dtype=np.uint8)
    min_dists = np.full((h_proc, w_proc), np.inf)

    print("Computing regions...")
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
    output_filename = "worley_pattern.png"
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
        
    generate_worley_mask(output_filename)
