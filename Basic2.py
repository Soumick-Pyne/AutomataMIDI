import numpy as np
import pygame
import fluidsynth
import time
import os
from enum import Enum
from typing import List, Tuple, Dict, NamedTuple
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from dataclasses import dataclass
import time
import random

class NoteLength(Enum):
    """Musical note lengths in terms of beats"""
    WHOLE = 4.0
    HALF = 2.0
    QUARTER = 1.0
    EIGHTH = 0.5
    SIXTEENTH = 0.25

class InstrumentConfig(NamedTuple):
    """Configuration for each instrument"""
    bank: int
    program: int
    display_name: str
    beat_pattern: List[int]  # List of beats (1-4) where this instrument can play
    gain: float = 0.5

class Instrument(Enum):
    """Configurable instruments with display names, beat patterns, and gains"""
    # Format: InstrumentConfig(bank, program, display_name, beat_pattern, gain)
    INSTRUMENT1 = InstrumentConfig(0, 0, "Acoustic Grand Piano", [1,2,3,4], 0.9)     
    INSTRUMENT2 = InstrumentConfig(0, 40, "Violin", [1,3], 0.8)                     
    INSTRUMENT3 = InstrumentConfig(0, 73, "Flute", [2,4], 0.3)                      
    INSTRUMENT4 = InstrumentConfig(0, 32, "Acoustic Bass", [1,2,3,4], 0.2)          

    @classmethod
    def set_instrument(cls, index: int, bank: int, program: int, display_name: str, 
                      beat_pattern: List[int], gain: float = 0.8):
        """Update instrument configuration"""
        instrument = list(cls)[index - 1]  # Convert 1-based index to 0-based
        instrument._value_ = InstrumentConfig(bank, program, display_name, beat_pattern, gain)
        return instrument

class ScaleType(Enum):
    C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]    # C major scale
    G_MAJOR = [67, 69, 71, 72, 74, 76, 78, 79]    # G major scale
    A_MINOR = [69, 71, 72, 74, 76, 77, 79, 81]    # A minor scale
    D_DORIAN = [62, 64, 65, 67, 69, 71, 72, 74]   # D Dorian mode
    E_PHRYGIAN = [64, 65, 67, 69, 71, 72, 74, 76] # E Phrygian mode

@dataclass
class MusicEvent:
    """Enhanced musical event with timing information"""
    note: int
    velocity: int
    channel: int
    duration: float
    start_offset: float  # Offset from the start of current update cycle
    note_length: float
    harmony_note: int = None
    scheduled_time: float = None

class PatternDetector:
    """Detects interesting patterns in the grid"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.visited = np.zeros((height, width), dtype=bool)

    def find_patterns(self, grid: np.ndarray) -> List[dict]:
        """Find interesting patterns in the grid"""
        self.visited.fill(False)
        patterns = []
        
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] and not self.visited[y, x]:
                    pattern = self._explore_pattern(grid, x, y)
                    if pattern['is_interesting']:
                        patterns.append(pattern)
        
        return patterns

    def _explore_pattern(self, grid: np.ndarray, start_x: int, start_y: int) -> dict:
        """Explore a pattern with enhanced musical awareness"""
        cells = []
        boundary = []
        empty_inside = []
        queue = [(start_x, start_y)]
        
        # Track pattern spans
        min_x = start_x
        max_x = start_x
        min_y = start_y
        max_y = start_y
        
        while queue:
            x, y = queue.pop(0)
            if self.visited[y, x]:
                continue
                
            self.visited[y, x] = True
            cells.append((x, y))
            
            # Update pattern spans
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            
            # Check neighbors with musical awareness
            has_empty_neighbor = False
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                
                if not grid[ny, nx]:
                    has_empty_neighbor = True
                    if self._is_surrounded_by_live_cells(grid, nx, ny):
                        empty_inside.append((nx, ny))
                elif not self.visited[ny, nx]:
                    queue.append((nx, ny))
            
            if has_empty_neighbor:
                boundary.append((x, y))
        
        # Calculate pattern characteristics
        horizontal_span = max_x - min_x + 1
        vertical_span = max_y - min_y + 1
        density = len(cells) / (horizontal_span * vertical_span)
        
        # Determine if pattern is musically interesting
        is_interesting = (
            len(cells) >= 3 and  # Minimum size for musical significance
            (len(empty_inside) > 0 or  # Has internal structure
             density > 0.6 or  # Dense enough to be significant
             (horizontal_span >= 3 and vertical_span >= 3))  # Large enough to be significant
        )
        
        return {
            'cells': cells,
            'boundary': boundary,
            'empty_inside': empty_inside,
            'size': len(cells),
            'is_interesting': is_interesting,
            'center': self._calculate_center(cells),
            'region': self._determine_region(cells[0][0], cells[0][1]),
            'shape_metrics': {
                'horizontal_span': horizontal_span,
                'vertical_span': vertical_span,
                'density': density
            }
        }

    def _is_surrounded_by_live_cells(self, grid: np.ndarray, x: int, y: int) -> bool:
        """Check if an empty cell is surrounded by live cells"""
        live_neighbors = 0
        total_checked = 0
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = (x + dx) % self.width, (y + dy) % self.height
            if grid[ny, nx]:
                live_neighbors += 1
            total_checked += 1
            
        return live_neighbors >= 6  # At least 6 live neighbors to consider "surrounded"

    def _calculate_center(self, cells: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate the center point of a pattern"""
        if not cells:
            return (0, 0)
        x_coords, y_coords = zip(*cells)
        return (int(sum(x_coords) / len(cells)), int(sum(y_coords) / len(cells)))

    def _determine_region(self, x: int, y: int) -> Tuple[int, int]:
        """Determine which quadrant/region a point belongs to"""
        region_x = 0 if x < self.width // 2 else 1
        region_y = 0 if y < self.height // 2 else 1
        return (region_x, region_y)

class MusicSystem:
    """Enhanced music system with proper timing"""
    def __init__(self, soundfont_path: str, update_interval: float):
        self.fs = fluidsynth.Synth(gain=2.0)
        self.fs.start(driver='dsound', midi_driver='none')
        
        sfid = self.fs.sfload(soundfont_path)
        if sfid == -1:
            raise RuntimeError("Failed to load soundfont")
        
        self.fs.sfont_select(0, sfid)
        self.event_queue = deque()
        
        # Derive tempo from update interval
        # update_interval is the time for one full bar/measure
        self.measure_length = 4  # 4 beats per measure
        self.tempo_bpm = (60.0 / update_interval) * 4  # Calculate tempo based on update interval
        self.beat_duration = update_interval / 4.0  # Each beat's duration
        self.cycle_duration = update_interval  # One cycle = one full measure = update_interval
        
        self.current_measure = 0
        self.current_beat = 0
        self.quantize_grid = 8  # Subdivide each beat into 8 parts
        
        # Initialize instruments
        for i, instrument in enumerate(Instrument):
            self.fs.program_select(i, sfid, instrument.value.bank, instrument.value.program)
            volume = min(127, int(instrument.value.gain * 127))
            self.fs.cc(i, 7, volume)
    
    def quantize_time(self, time_offset: float) -> float:
        """Quantize a time offset to the nearest rhythmic grid position"""
        grid_duration = self.beat_duration / self.quantize_grid
        quantized_steps = round(time_offset / grid_duration)
        return quantized_steps * grid_duration
    
    def schedule_notes_for_update(self, events: List[MusicEvent]):
        """Schedule notes with musical timing"""
        if not events:
            return
            
        current_time = time.time()
        if (not self.event_queue or 
            current_time > self.update_cycle_start + self.cycle_duration):
            self.update_cycle_start = current_time
            self.current_beat = (self.current_beat + 1) % self.measure_length
            if self.current_beat == 0:
                self.current_measure += 1
        
        # Group events by channel/instrument
        events_by_channel = {}
        for event in events:
            if event.channel not in events_by_channel:
                events_by_channel[event.channel] = []
            events_by_channel[event.channel].append(event)
        
        # Calculate musical positions for each event
        for channel, channel_events in events_by_channel.items():
            instrument = list(Instrument)[channel]
            allowed_beats = [beat - 1 for beat in instrument.value.beat_pattern]  # Convert 1-based to 0-based beats
            
            # Only schedule up to 4 events, one per allowed beat
            events_scheduled = 0
            for beat in allowed_beats:
                if events_scheduled >= len(channel_events):
                    break
                    
                event = channel_events[events_scheduled]
                position_in_measure = beat  # Position based on the beat number
                quantized_position = self.quantize_time(position_in_measure * self.beat_duration)
                event.start_offset = quantized_position
                
                # Add slight humanization (+/- 30ms)
                humanize_offset = random.uniform(-0.03, 0.03)
                event.start_offset += humanize_offset
                
                # Schedule the event
                event.scheduled_time = self.update_cycle_start + event.start_offset
                self.event_queue.append(event)
                events_scheduled += 1
        
        # Sort queue by scheduled time
        events_list = list(self.event_queue)
        events_list.sort(key=lambda x: x.scheduled_time)
        self.event_queue = deque(events_list)

    def update(self):
        """Process scheduled music events with proper timing"""
        current_time = time.time()
        
        # Process all events that are due
        while self.event_queue:
            event = self.event_queue[0]
            event_time = self.update_cycle_start + event.start_offset
            
            if current_time >= event_time:
                # Time to play this note
                self.event_queue.popleft()
                
                # Play the note
                self.fs.noteon(event.channel, event.note, event.velocity)
                if event.harmony_note is not None:
                    self.fs.noteon(event.channel, event.harmony_note, event.velocity - 10)
                
                # Schedule note off based on note length
                note_duration = event.note_length * self.beat_duration
                custom_event = pygame.event.Event(
                    pygame.USEREVENT + event.channel,
                    {
                        'note': event.note,
                        'harmony_note': event.harmony_note,
                        'scheduled_time': current_time + note_duration
                    }
                )
                pygame.event.post(custom_event)
            else:
                # If this event isn't ready, no later events will be ready
                break
    
    def handle_note_off(self, channel: int, note: int, harmony_note: int = None):
        """Handle note off events based on scheduled timing"""
        current_time = time.time()
        
        # Turn off main note
        self.fs.noteoff(channel, note)
        
        # Turn off harmony note if present
        if harmony_note is not None:
            self.fs.noteoff(channel, harmony_note)

class GameState:
    """Manages the game state separately from music generation"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.current_grid = np.zeros((height, width), dtype=bool)
        self.next_grid = np.zeros((height, width), dtype=bool)
        self.last_update = time.time()
        self.update_interval = 1.0

    def should_update(self) -> bool:
        """Check if it's time for a state update"""
        current_time = time.time()
        should_update = current_time - self.last_update >= self.update_interval
        return should_update

    def update(self) -> List[Tuple[int, int]]:
        """Update the game state and return list of cells that changed state"""
        changed_cells = []
        
        # Calculate next state for all cells
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self._count_neighbors(x, y)
                current_state = self.current_grid[y, x]
                
                # Apply Conway's rules
                if current_state:
                    self.next_grid[y, x] = neighbors in [2, 3]
                else:
                    self.next_grid[y, x] = neighbors == 3
                
                # Track changed cells
                if self.next_grid[y, x] != current_state:
                    changed_cells.append((x, y))

        # Update current grid
        self.current_grid = self.next_grid.copy()
        self.next_grid = np.zeros((self.height, self.width), dtype=bool)  # Reset next grid
        self.last_update = time.time()
        return changed_cells

    def _count_neighbors(self, x: int, y: int) -> int:
        """Count living neighbors for a cell"""
        total = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                if self.current_grid[ny, nx]:
                    total += 1
        return total

class MusicalAutomata:
    def __init__(self, width: int = 32, height: int = 32):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.cell_size = 20
        
        # Initialize game state first to get update interval
        self.game_state = GameState(width, height)
        
        # Initialize music system with game state's update interval
        self.music_system = MusicSystem(
            r"D:\\Music\\MusicAutomata\\FluidR3_GM\\FluidR3_GM.sf2",
            update_interval=self.game_state.update_interval
        )

        # Pattern detector
        self.pattern_detector = PatternDetector(width, height)
        
        # Setup display with additional height for instrument labels
        self.label_height = 30
        screen_width = width * self.cell_size
        screen_height = height * self.cell_size + self.label_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Musical Cellular Automata")
        
        # Font setup
        self.font = pygame.font.SysFont('Arial', 14)
        
        # Grid regions setup
        self.regions = self._setup_grid_regions()
        
        # Visual properties with instrument-specific colors
        self.COLORS = {
            Instrument.INSTRUMENT1: (200, 200, 255),
            Instrument.INSTRUMENT2: (255, 200, 200),
            Instrument.INSTRUMENT3: (200, 255, 200),
            Instrument.INSTRUMENT4: (255, 255, 200)
        }
        self.BACKGROUND = (10, 10, 10)
        self.CELL_DEAD = (40, 40, 40)

    def _setup_grid_regions(self) -> Dict:
        """Setup grid regions with their musical properties"""
        regions = {}
        half_width = self.width // 2
        half_height = self.height // 2
        
        # Define regions with their properties
        region_props = [
            ((0, 0), (half_width, half_height), 
            Instrument.INSTRUMENT1, ScaleType.C_MAJOR),
            ((half_width, 0), (self.width, half_height), 
            Instrument.INSTRUMENT2, ScaleType.G_MAJOR),
            ((0, half_height), (half_width, self.height), 
            Instrument.INSTRUMENT3, ScaleType.A_MINOR),
            ((half_width, half_height), (self.width, self.height), 
            Instrument.INSTRUMENT4, ScaleType.D_DORIAN)
        ]
        
        for (x1, y1), (x2, y2), instrument, scale in region_props:
            for x in range(x1, x2):
                for y in range(y1, y2):
                    regions[(x, y)] = {
                        'instrument': instrument,
                        'scale': scale,
                        'channel': list(Instrument).index(instrument)
                    }
        
        return regions

    def _create_music_event(self, pattern: dict) -> MusicEvent:
        """Create a more musically coherent event based on pattern"""
        center_x, center_y = pattern['center']
        region = self.regions[(center_x, center_y)]
        scale = region['scale'].value
        
        # Base velocity on pattern size but with more musical dynamics
        base_velocity = 60  # Softer base velocity
        size_boost = min(40, pattern['size'] * 3)
        velocity = min(127, base_velocity + size_boost)
        
        # Choose note based on pattern characteristics
        position_in_scale = len(pattern['cells']) % len(scale)
        base_note = scale[position_in_scale]
        
        # Add harmony based on musical rules
        harmony_note = None
        if pattern['size'] > 3:
            if len(pattern['empty_inside']) > 0:
                # Use thirds for patterns with holes
                harmony_note = base_note + (4 if random.random() > 0.5 else 3)
            else:
                # Use fifths for solid patterns
                harmony_note = base_note + 7
        
        # Create main event with fixed quarter note duration
        return MusicEvent(
            note=base_note,
            velocity=velocity,
            channel=region['channel'],
            duration=1.0,  # One beat duration
            start_offset=0.0,
            note_length=1.0,  # quarter note duration
            harmony_note=harmony_note
        )

    def update(self):
        """Enhanced main update loop with pattern-based music generation"""
        if self.game_state.should_update():
            changed_cells = self.game_state.update()
            
            if changed_cells:  # Only process if there were changes
                # Find patterns in the current grid
                patterns = self.pattern_detector.find_patterns(self.game_state.current_grid)
                
                # Generate events for interesting patterns
                events = []
                for pattern in patterns:
                    event = self._create_music_event(pattern)
                    events.append(event)
                
                # Schedule all events together
                if events:
                    self.music_system.schedule_notes_for_update(events)
        
        # Update music system
        self.music_system.update()

    def draw(self):
        """Draw the current state with instrument labels"""
        self.screen.fill(self.BACKGROUND)
        
        # Draw grid cells
        for y in range(self.height):
            for x in range(self.width):
                region = self.regions[(x, y)]
                color = (self.COLORS[region['instrument']] 
                        if self.game_state.current_grid[y, x] 
                        else self.CELL_DEAD)
                
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size + self.label_height,  # Offset for labels
                    self.cell_size - 1, 
                    self.cell_size - 1
                )
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw region dividers
        mid_x = (self.width // 2) * self.cell_size
        mid_y = (self.height // 2) * self.cell_size + self.label_height
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (mid_x, self.label_height), 
                        (mid_x, self.height * self.cell_size + self.label_height))
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (0, mid_y), 
                        (self.width * self.cell_size, mid_y))
        
        # Draw instrument labels
        quarter_width = self.width * self.cell_size // 2
        instruments = list(Instrument)
        label_positions = [
            (quarter_width // 2, 5),                    # Top left
            (quarter_width * 3 // 2, 5),               # Top right
            (quarter_width // 2, self.label_height + mid_y - 25),     # Bottom left
            (quarter_width * 3 // 2, self.label_height + mid_y - 25)  # Bottom right
        ]
        
        for instrument, pos in zip(instruments, label_positions):
            label = self.font.render(instrument.value.display_name, True, self.COLORS[instrument])
            label_rect = label.get_rect(center=pos)
            self.screen.blit(label, label_rect)
        
        pygame.display.flip()

    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Randomize grid
                        self.game_state.current_grid = np.random.random(
                            (self.height, self.width)) > 0.85
                elif event.type >= pygame.USEREVENT:
                    # Handle note off events
                    channel = event.type - pygame.USEREVENT
                    self.music_system.handle_note_off(channel, event.dict['note'], 
                                                    event.dict.get('harmony_note'))
            
            # Handle mouse input
            if pygame.mouse.get_pressed()[0]:
                x, y = pygame.mouse.get_pos()
                grid_x, grid_y = x // self.cell_size, y // self.cell_size - self.label_height // self.cell_size
                if (0 <= grid_x < self.width and 0 <= grid_y < self.height and
                    not self.game_state.current_grid[grid_y, grid_x]):
                    # Set cell to alive
                    self.game_state.current_grid[grid_y, grid_x] = True
                    
                    # Find patterns that include this cell
                    patterns = self.pattern_detector.find_patterns(self.game_state.current_grid)
                    
                    # Create events for any interesting patterns
                    events = []
                    for pattern in patterns:
                        if (grid_x, grid_y) in pattern['cells']:  # Only play if our new cell is part of a pattern
                            event = self._create_music_event(pattern)
                            events.append(event)
                    
                    # Schedule events
                    if events:
                        self.music_system.schedule_notes_for_update(events)
            
            # Update and draw
            self.update()
            self.draw()
            clock.tick(60)
        
        pygame.quit()

def set_instrument_config(index: int, bank: int, program: int, display_name: str, beat_pattern: List[int], gain: float = 0.8):
    """Utility function to configure instruments before creating the automata"""
    return Instrument.set_instrument(index, bank, program, display_name, beat_pattern, gain)

if __name__ == "__main__":
    try:
        set_instrument_config(1, 0, 95, "Halo Pad", [1,2], 0.5)           # Heavenly pad
        set_instrument_config(2, 0, 99, "Crystal", [3,4], 0.3)            # Sparkling textures
        set_instrument_config(3, 0, 88, "Bass+Lead", [1], 0.4)            # Strong accents
        set_instrument_config(4, 0, 92, "Choir Pad", [1,3], 0.2)          # Vocal textures

        automata = MusicalAutomata()
        automata.run()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")