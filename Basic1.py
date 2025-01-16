import numpy as np
import pygame
import fluidsynth
import time
import os

class MusicCell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.alive = False
        self.note = 60  # Middle C by default
        self.velocity = 80
        self.last_played = 0  # To prevent too frequent playing

class MusicalAutomata:
    def __init__(self, width: int = 32, height: int = 32):
        # Initialize Pygame
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = 20
        screen_width = width * self.cell_size
        screen_height = height * self.cell_size
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Musical Cellular Automata")
        
        # Create grid of cells
        self.grid = [[MusicCell(x, y) for x in range(width)] for y in range(height)]
        
        # Initialize FluidSynth with explicit settings
        print("Initializing FluidSynth...")
        self.fs = fluidsynth.Synth(gain=0.5)
        
        # Start with explicit settings
        self.fs.start(driver='dsound', midi_driver='none')  # Explicitly disable MIDI input
        
        # Load soundfont
        soundfont_path = r"D:\\Music\\MusicAutomata\\FluidR3_GM\\FluidR3_GM.sf2"
        print(f"Loading soundfont: {soundfont_path}")
        sfid = self.fs.sfload(soundfont_path)
        if sfid == -1:
            raise RuntimeError("Failed to load soundfont")
            
        self.fs.sfont_select(0, sfid)
        self.fs.program_select(0, sfid, 0, 0)  # Piano
        
        # Musical properties
        self.scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
        self.last_note_time = time.time()
        self.min_note_interval = 0.2  # Minimum time between notes
        
        # Colors
        self.BACKGROUND = (10, 10, 10)
        self.CELL_DEAD = (40, 40, 40)
        self.CELL_ALIVE = (200, 200, 255)
        
        print("Initialization complete!")
    
    def play_note(self, note, velocity=80):
        current_time = time.time()
        if current_time - self.last_note_time >= self.min_note_interval:
            try:
                self.fs.noteon(0, note, velocity)
                time.sleep(0.1)  # Hold note briefly
                self.fs.noteoff(0, note)
                self.last_note_time = current_time
            except Exception as e:
                print(f"Note playing error: {e}")
    
    def get_neighbors(self, x: int, y: int):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x = (x + dx) % self.width
                new_y = (y + dy) % self.height
                neighbors.append(self.grid[new_y][new_x])
        return neighbors
    
    def update_cells(self):
        # First calculate all next states without updating
        to_update = []
        next_states = {}  # Store next states before applying them
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                neighbors = self.get_neighbors(x, y)
                living_neighbors = sum(1 for n in neighbors if n.alive)
                
                # Calculate next state without updating immediately
                if cell.alive:
                    next_states[(x, y)] = living_neighbors in [2, 3]
                else:
                    next_states[(x, y)] = living_neighbors == 3
                
                # If state will change, add to update list
                if next_states[(x, y)] != cell.alive:
                    to_update.append((cell, next_states[(x, y)]))
        
        # Then update all states and play notes
        current_time = time.time()
        for cell, new_state in to_update:
            cell.alive = new_state
            if cell.alive and current_time - cell.last_played >= self.min_note_interval:
                self.play_note(np.random.choice(self.scale))
                cell.last_played = current_time
    
    def draw(self):
        self.screen.fill(self.BACKGROUND)
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                color = self.CELL_ALIVE if cell.alive else self.CELL_DEAD
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size - 1, self.cell_size - 1)
                pygame.draw.rect(self.screen, color, rect)
        
        pygame.display.flip()
    
    def handle_mouse_input(self):
        if pygame.mouse.get_pressed()[0]:  # Left mouse button
            x, y = pygame.mouse.get_pos()
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                cell = self.grid[grid_y][grid_x]
                if not cell.alive:  # Only toggle if cell was dead
                    cell.alive = True
                    self.play_note(np.random.choice(self.scale))
                    self.draw()  # Immediately update the display
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Randomize grid
                    for row in self.grid:
                        for cell in row:
                            cell.alive = np.random.random() > 0.85
                            cell.next_state = cell.alive
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        self.handle_mouse_input()
        return True
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        update_interval = 0.2  # Update cells every 200ms
        last_update = time.time()
        
        print("\nControls:")
        print("- Click and drag to create living cells")
        print("- Press SPACE to randomize the grid")
        print("- Press ESC to quit")
        
        while running:
            running = self.handle_events()
            
            # Only update cells at fixed interval
            current_time = time.time()
            if current_time - last_update >= update_interval:
                self.update_cells()
                last_update = current_time
                
            self.draw()
            clock.tick(60)  # Higher frame rate for smooth mouse interaction
        
        self.fs.delete()
        pygame.quit()

if __name__ == "__main__":
    try:
        automata = MusicalAutomata()
        automata.run()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")