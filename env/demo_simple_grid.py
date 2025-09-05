#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 当家
# @time    : 2025/7/4 8:00
# @function: Simple demo for grid visualization
# @version : V1

from GridVisualizer import GridVisualizer, CellType, ActionType
import matplotlib.pyplot as plt

def create_simple_demo():
    """Create a simple 3x3 grid demonstration"""
    # Create 3x3 grid
    grid = GridVisualizer(3, 3)
    
    print("Creating 3x3 Grid Demo...")
    print("State numbering (left to right, top to bottom):")
    for i in range(3):
        row_states = []
        for j in range(3):
            state_num = grid.get_state_number(i, j)
            row_states.append(f"s{state_num}")
        print(f"Row {i}: {' '.join(row_states)}")
    
    # Set target at bottom-right
    grid.set_cell_type(2, 2, CellType.TARGET)
    print("\nSet s9 (bottom-right) as TARGET (blue)")
    
    # Set forbidden in center
    grid.set_cell_type(1, 1, CellType.FORBIDDEN)
    print("Set s5 (center) as FORBIDDEN (yellow)")
    
    # Set actions to move towards target
    grid.set_action(0, 0, ActionType.RIGHT)  # s1: move right
    grid.set_action(0, 1, ActionType.DOWN)   # s2: move down
    grid.set_action(0, 2, ActionType.DOWN)   # s3: move down
    grid.set_action(1, 0, ActionType.RIGHT)  # s4: move right
    # s5 is forbidden - keep default STAY (circle)
    grid.set_action(1, 2, ActionType.DOWN)   # s6: move down
    grid.set_action(2, 0, ActionType.RIGHT)  # s7: move right
    grid.set_action(2, 1, ActionType.RIGHT)  # s8: move right
    grid.set_action(2, 2, ActionType.STAY)   # s9: stay at target
    
    print("\nActions set:")
    print("s1: a2(→), s2: a3(↓), s3: a3(↓)")
    print("s4: a2(→), s5: a5(○), s6: a3(↓)")
    print("s7: a2(→), s8: a2(→), s9: a5(○)")
    
    return grid

if __name__ == '__main__':
    print('Simple Grid Visualization Demo')
    print('=' * 40)
    
    # Create simple demo
    grid = create_simple_demo()
    
    # Visualize
    fig, ax = grid.visualize(figsize=(8, 6))
    
    # Save the plot
    plt.savefig('/tmp/simple_grid_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nSimple demo saved to /tmp/simple_grid_demo.png")
    
    print(f"\nGrid Summary:")
    print(f"- Size: 3×3 grid")
    print(f"- States: s1 to s9 (numbered left→right, top→bottom)")
    print(f"- s9 (bottom-right): TARGET (blue)")
    print(f"- s5 (center): FORBIDDEN (yellow)")
    print(f"- All others: NORMAL (white)")
    print(f"- Actions show paths toward target")