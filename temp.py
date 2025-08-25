import kaggle_environments
from kaggle_environments import make, utils
import numpy as np

submission = utils.read_file("opponents/submission_v47.py")
agent = utils.get_last_callable(submission)

l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]
print(type(l))
print(len(l))
print(l)
board = np.array(l)
print(type(board))
config = {"rows": 6, "columns": 7, "inarow": 4}
temp = {
    'board': board,
    'mark' : 1
}
print(board)
action = agent(temp, config)
print(action)
# import os
# f = os.listdir(os.path.join('opponents'))
# print(f)