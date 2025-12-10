import numpy as np
from PIL import Image, ImageDraw
import random
import sys

WIDTH = 800
HEIGHT = 480
GRID_ROWS = 20         
GRID_COLS = 30         
JITTER_AMOUNT = 0.75    
BLOCK_RATIO = 0.20

HIGHLIGHT_CHANCE = 0.50  # 15% chance a color block is bright, 85% chance it's dim

DIM_RANGE = (50, 100)     # RGB values for the "dim" color blocks
BRIGHT_RANGE = (150, 255) # RGB values for the "bright" color blocks

DARK_MODE = True       
ANCHOR_BORDER = True   

def generate_grid_mask(filename):
    num_cells = GRID_ROWS * GRID_COLS
    print(f"Generating {GRID_ROWS}x{GRID_COLS} grid")
    print(f"Stats: {BLOCK_RATIO*100}% Blockers | Of the rest: {HIGHLIGHT_CHANCE*100}% Bright, {(1-HIGHLIGHT_CHANCE)*100}% Dim")

    w_proc = WIDTH // 4
    h_proc = HEIGHT // 4
    
    cell_w = w_proc / GRID_COLS
    cell_h = h_proc / GRID_ROWS

    seeds_x = []
    seeds_y = []
    seed_colors = []

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            base_x = c * cell_w
            base_y = r * cell_h
            margin_w = cell_w * (1 - JITTER_AMOUNT) / 2
            margin_h = cell_h * (1 - JITTER_AMOUNT) / 2
            jx = random.uniform(margin_w, cell_w - margin_w)
            jy = random.uniform(margin_h, cell_h - margin_h)
            
            seeds_x.append(int(base_x + jx))
            seeds_y.append(int(base_y + jy))
            
            if random.random() < BLOCK_RATIO:
                if DARK_MODE:
                    r_val = random.choice([0, 5, 8])
                    g_val = random.choice([0, 5, 8])
                    b_val = random.choice([0, 5, 8])
                    seed_colors.append((r_val, g_val, b_val))
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

    seeds_x = np.array(seeds_x)
    seeds_y = np.array(seeds_y)

    y, x = np.indices((h_proc, w_proc))
    output_buffer = np.zeros((h_proc, w_proc, 3), dtype=np.uint8)
    min_dists = np.full((h_proc, w_proc), np.inf)

    print("Computing regions...")
    for i in range(num_cells):
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
    output_filename = "grid_biased.png"
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
    generate_grid_mask(output_filename)
