#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : Zhang Jinbin
# @time    : 2025/7/24 12:14
# @function: the script is used to do something.
# @version : V1


import queue
import threading
import time
import logging


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../app.log'
)

# 使用
logger = logging.getLogger("my_app")
logger.info("程序启动")

# 创建一个队列
q = queue.Queue()


# 发送者线程
def sender():
    for i in range(5):
        time.sleep(1)
        q.put(i)
        print('send:', i)


# 接收者线程
def receiver():
    while True:
        item = q.get()
        if item is None:
            break
        print(f'recv:{item}')
        q.task_done()


sender_thread = threading.Thread(target=sender)
receiver_thread = threading.Thread(target=receiver)

# 开启线程
sender_thread.start()
receiver_thread.start()

sender_thread.join()

# 等待队列中的所有任务完成
q.join()

# 发送结束信号
q.put(None)
receiver_thread.join()
