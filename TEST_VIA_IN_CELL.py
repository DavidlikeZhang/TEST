#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 当家
# @time    : 2025/7/4 7:44
# @function: the script is used to do something.
# @version : V1

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('TEST_VIA_IN_CELL')

    np.random.seed(123)
    x = np.random.rand(100)
    y = np.random.rand(100)#
    colors = np.random.rand(100)
    area = (30 * np.random.rand(100))**2  # 0 to 15 point radii
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()