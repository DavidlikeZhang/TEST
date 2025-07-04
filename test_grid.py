#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 当家
# @time    : 2025/7/4 8:00
# @function: Tests for grid visualization
# @version : V1

import sys
import os
sys.path.append(os.path.dirname(__file__))

from TEST_VIA_IN_CELL import GridVisualizer, CellType, ActionType

def test_grid_creation():
    """Test basic grid creation"""
    print("Testing grid creation...")
    grid = GridVisualizer(3, 4)
    assert grid.m == 3
    assert grid.n == 4
    print("✓ Grid creation test passed")

def test_state_numbering():
    """Test state numbering logic"""
    print("Testing state numbering...")
    grid = GridVisualizer(3, 3)
    
    # Test state numbering (left to right, top to bottom)
    expected_states = [
        [1, 2, 3],
        [4, 5, 6], 
        [7, 8, 9]
    ]
    
    for i in range(3):
        for j in range(3):
            state_num = grid.get_state_number(i, j)
            assert state_num == expected_states[i][j], f"Expected {expected_states[i][j]}, got {state_num}"
    
    print("✓ State numbering test passed")

def test_cell_types():
    """Test setting cell types"""
    print("Testing cell types...")
    grid = GridVisualizer(2, 2)
    
    # Test setting different cell types
    grid.set_cell_type(0, 0, CellType.TARGET)
    grid.set_cell_type(0, 1, CellType.FORBIDDEN)
    # (1,0) and (1,1) remain NORMAL
    
    assert grid.cell_types[0, 0] == CellType.TARGET
    assert grid.cell_types[0, 1] == CellType.FORBIDDEN
    assert grid.cell_types[1, 0] == CellType.NORMAL
    assert grid.cell_types[1, 1] == CellType.NORMAL
    
    print("✓ Cell types test passed")

def test_actions():
    """Test setting actions"""
    print("Testing actions...")
    grid = GridVisualizer(2, 2)
    
    # Test setting different actions
    grid.set_action(0, 0, ActionType.UP)
    grid.set_action(0, 1, ActionType.RIGHT)
    grid.set_action(1, 0, ActionType.DOWN)
    grid.set_action(1, 1, ActionType.LEFT)
    
    assert grid.actions[0, 0] == ActionType.UP
    assert grid.actions[0, 1] == ActionType.RIGHT
    assert grid.actions[1, 0] == ActionType.DOWN
    assert grid.actions[1, 1] == ActionType.LEFT
    
    print("✓ Actions test passed")

def test_bounds_checking():
    """Test bounds checking for invalid positions"""
    print("Testing bounds checking...")
    grid = GridVisualizer(2, 2)
    
    # Test invalid positions
    try:
        grid.set_cell_type(2, 0, CellType.TARGET)  # Row out of bounds
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        grid.set_action(0, 2, ActionType.UP)  # Column out of bounds
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ Bounds checking test passed")

def test_visualization():
    """Test visualization creation (doesn't crash)"""
    print("Testing visualization...")
    grid = GridVisualizer(2, 2)
    
    # Set up a simple grid
    grid.set_cell_type(0, 0, CellType.TARGET)
    grid.set_action(0, 1, ActionType.RIGHT)
    
    # This should not crash
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        fig, ax = grid.visualize()
        print("✓ Visualization test passed")
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all tests"""
    print("Running Grid Visualization Tests")
    print("=" * 40)
    
    try:
        test_grid_creation()
        test_state_numbering()
        test_cell_types()
        test_actions()
        test_bounds_checking()
        test_visualization()
        
        print("\n" + "=" * 40)
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)