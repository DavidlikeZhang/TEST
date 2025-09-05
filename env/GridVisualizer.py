#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 当家
# @time    : 2025/7/4 7:44
# @function: Grid visualization with states and actions
# @version : V2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
import os


class CellType(Enum):
    """Cell types for the grid"""
    NORMAL = 'white'      # Normal cell (white)
    TARGET = 'blue'       # Target cell (blue)
    FORBIDDEN = 'yellow'  # Forbidden area (yellow)


class ActionType(Enum):
    """Action types for each state"""
    UP = 1     # a1 - up arrow
    RIGHT = 2  # a2 - right arrow
    DOWN = 3   # a3 - down arrow
    LEFT = 4   # a4 - left arrow
    STAY = 5   # a5 - circle (stay in place)

class GridVisualizer:
    """Grid visualization class for states and actions"""
    
    def __init__(self, m, n):
        """
        Initialize grid visualizer
        
        Args:
            m (int): Number of rows
            n (int): Number of columns
        """
        self.m = m
        self.n = n
        self.cell_types = np.full((m, n), CellType.NORMAL)
        self.actions = np.full((m, n), ActionType.STAY, dtype=object)
        
    def set_cell_type(self, row, col, cell_type):
        """Set cell type at given position"""
        if 0 <= row < self.m and 0 <= col < self.n:
            self.cell_types[row, col] = cell_type
        else:
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
    
    def set_action(self, row, col, action):
        """Set action at given position"""
        if 0 <= row < self.m and 0 <= col < self.n:
            self.actions[row, col] = action
        else:
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
    
    def get_state_number(self, row, col):
        """Get state number for given position (left to right, top to bottom numbering)"""
        return row * self.n + col + 1
    
    def visualize(self, figsize=(10, 8), show_state_numbers=True, show_actions=True):
        """
        Visualize the grid with states and actions
        
        Args:
            figsize (tuple): Figure size
            show_state_numbers (bool): Whether to show state numbers
            show_actions (bool): Whether to show actions
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up the grid
        ax.set_xlim(-0.5, self.n - 0.5)
        ax.set_ylim(-0.5, self.m - 0.5)
        ax.set_aspect('equal')
        
        # Invert y-axis so (0,0) is at top-left
        ax.invert_yaxis()
        
        # Draw grid cells
        for i in range(self.m):
            for j in range(self.n):
                # Get cell color
                cell_type = self.cell_types[i, j]
                color = cell_type.value
                
                # Create rectangle for cell
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Add state number
                if show_state_numbers:
                    state_num = self.get_state_number(i, j)
                    ax.text(j, i-0.3, f's{state_num}', 
                           ha='center', va='center', fontsize=10, fontweight='bold')
                
                # Add action visualization
                if show_actions:
                    action = self.actions[i, j]
                    self._draw_action(ax, i, j, action)
        
        # Set grid lines
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.m))
        ax.grid(True, alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title(f'{self.m}×{self.n} Grid State Visualization', fontsize=14, fontweight='bold')
        
        # Legend
        legend_elements = [
            patches.Patch(color='white', label='Normal'),
            patches.Patch(color='blue', alpha=0.7, label='Target'),
            patches.Patch(color='yellow', alpha=0.7, label='Forbidden')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        return fig, ax
    
    def _draw_action(self, ax, row, col, action):
        """Draw action arrow or circle at given position"""
        x, y = col, row + 0.1  # Slight offset to avoid overlap with state number
        
        if action == ActionType.UP:
            ax.annotate('', xy=(x, y-0.2), xytext=(x, y+0.1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            ax.text(x+0.25, y, 'a1', ha='center', va='center', fontsize=8, color='red')
        elif action == ActionType.RIGHT:
            ax.annotate('', xy=(x+0.2, y), xytext=(x-0.1, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            ax.text(x, y+0.25, 'a2', ha='center', va='center', fontsize=8, color='red')
        elif action == ActionType.DOWN:
            ax.annotate('', xy=(x, y+0.2), xytext=(x, y-0.1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            ax.text(x-0.25, y, 'a3', ha='center', va='center', fontsize=8, color='red')
        elif action == ActionType.LEFT:
            ax.annotate('', xy=(x-0.2, y), xytext=(x+0.1, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            ax.text(x, y-0.25, 'a4', ha='center', va='center', fontsize=8, color='red')
        elif action == ActionType.STAY:
            circle = patches.Circle((x, y), 0.15, linewidth=2, 
                                  edgecolor='red', facecolor='none')
            ax.add_patch(circle)
            ax.text(x+0.25, y-0.25, 'a5', ha='center', va='center', fontsize=8, color='red')

    def set_all_action(self, policy):
        self.actions = policy

def create_example_grid():
    """Create an example grid for demonstration"""
    # Create 4x5 grid
    grid = GridVisualizer(4, 5)
    
    # Set some target cells (blue)
    grid.set_cell_type(0, 4, CellType.TARGET)  # Top-right corner
    
    # Set some forbidden areas (yellow)
    grid.set_cell_type(1, 2, CellType.FORBIDDEN)
    grid.set_cell_type(2, 2, CellType.FORBIDDEN)
    grid.set_cell_type(2, 3, CellType.FORBIDDEN)
    
    # Set various actions
    grid.set_action(0, 0, ActionType.RIGHT)  # s1: right arrow
    grid.set_action(0, 1, ActionType.RIGHT)  # s2: right arrow
    grid.set_action(0, 2, ActionType.DOWN)   # s3: down arrow
    grid.set_action(0, 3, ActionType.RIGHT)  # s4: right arrow
    grid.set_action(0, 4, ActionType.STAY)   # s5: stay (target)
    
    grid.set_action(1, 0, ActionType.UP)     # s6: up arrow
    grid.set_action(1, 1, ActionType.RIGHT)  # s7: right arrow
    # s8 (1,2) is forbidden - keep default STAY
    grid.set_action(1, 3, ActionType.DOWN)   # s9: down arrow
    grid.set_action(1, 4, ActionType.LEFT)   # s10: left arrow
    
    grid.set_action(2, 0, ActionType.UP)     # s11: up arrow
    grid.set_action(2, 1, ActionType.UP)     # s12: up arrow
    # s13, s14 (2,2), (2,3) are forbidden - keep default STAY
    grid.set_action(2, 4, ActionType.LEFT)   # s15: left arrow
    
    grid.set_action(3, 0, ActionType.STAY)   # s16: stay (target)
    grid.set_action(3, 1, ActionType.LEFT)   # s17: left arrow
    grid.set_action(3, 2, ActionType.UP)     # s18: up arrow
    grid.set_action(3, 3, ActionType.UP)     # s19: up arrow
    grid.set_action(3, 4, ActionType.UP)     # s20: up arrow
    
    return grid

if __name__ == '__main__':
    print('Grid Visualization with States and Actions')
    
    # Create and visualize example grid
    grid = create_example_grid()
    fig, ax = grid.visualize()
    
    # Save the plot
    try:
        os.chdir("../img")
    except OSError:
        os .mkdir("../img")
        os.chdir("../img")
    plt.savefig('grid_visualization.png', dpi=150, bbox_inches='tight')
    print("Grid visualization saved to /tmp/grid_visualization.png")

    # Also show if display is available
    try:
        plt.show()
    except:
        print("Display not available, plot saved to file instead")
    
    # Print some information
    print(f"\nGrid Information:")
    print(f"Size: {grid.m}×{grid.n}")
    print(f"States: s1 to s{grid.m * grid.n}")
    print(f"Actions: a1(↑), a2(→), a3(↓), a4(←), a5(○)")
    print(f"Cell types: Normal(white), Target(blue), Forbidden(yellow)")