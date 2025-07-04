# Grid Visualization Program

This program creates m×n grid visualizations with states and actions using Python matplotlib and numpy.

## Features

- **Grid Creation**: Create m×n grids with customizable dimensions
- **Cell Types**: 
  - Normal cells (white)
  - Target cells (blue) 
  - Forbidden areas (yellow)
- **State Numbering**: States numbered s1, s2, ... from left to right, top to bottom
- **Action Visualization**: 
  - a1: Up arrow (↑)
  - a2: Right arrow (→)
  - a3: Down arrow (↓)
  - a4: Left arrow (←)
  - a5: Circle (○) - stay in place

## Usage

### Basic Example

```python
from TEST_VIA_IN_CELL import GridVisualizer, CellType, ActionType
import matplotlib.pyplot as plt

# Create a 3x3 grid
grid = GridVisualizer(3, 3)

# Set cell types
grid.set_cell_type(0, 2, CellType.TARGET)     # Top-right as target
grid.set_cell_type(1, 1, CellType.FORBIDDEN)  # Center as forbidden

# Set actions
grid.set_action(0, 0, ActionType.RIGHT)  # s1: move right
grid.set_action(0, 1, ActionType.DOWN)   # s2: move down

# Visualize
fig, ax = grid.visualize()
plt.show()
```

### State Numbering

States are numbered from 1, going left to right, top to bottom:

```
3x3 Grid:
s1  s2  s3
s4  s5  s6  
s7  s8  s9
```

### Available Cell Types

- `CellType.NORMAL`: White cells (default)
- `CellType.TARGET`: Blue cells
- `CellType.FORBIDDEN`: Yellow cells

### Available Actions

- `ActionType.UP`: a1 - Up arrow
- `ActionType.RIGHT`: a2 - Right arrow  
- `ActionType.DOWN`: a3 - Down arrow
- `ActionType.LEFT`: a4 - Left arrow
- `ActionType.STAY`: a5 - Circle (default)

## Files

- `TEST_VIA_IN_CELL.py`: Main implementation with GridVisualizer class
- `demo_simple_grid.py`: Simple 3x3 demonstration
- `README.md`: This documentation

## Requirements

- Python 3.x
- numpy
- matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Examples

### Run the main example (4x5 grid):
```bash
python3 TEST_VIA_IN_CELL.py
```

### Run the simple demo (3x3 grid):
```bash
python3 demo_simple_grid.py
```

Both will save PNG files to `/tmp/` directory for viewing.