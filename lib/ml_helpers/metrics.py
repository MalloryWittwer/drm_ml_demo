import tensorflow as tf
from lib.utils import rot_mat_to_trace, eulers_to_rot_mat

SYMMETRIES = tf.Variable([
    [[1,0,0],[0,1,0],[0,0,1]],
    [[0,0,1],[1,0,0],[0,1,0]],
    [[0,1,0],[0,0,1],[1,0,0]], 
    [[0,-1,0],[0,0,1],[-1,0,0]], 
    [[0,-1,0],[0,0,-1],[1,0,0]], 
    [[0,1,0],[0,0,-1],[-1,0,0]],
    [[0,0,-1],[1,0,0],[0,-1,0]],
    [[0,0,-1],[-1,0,0],[0,1,0]],
    [[0,0,1],[-1,0,0],[0,-1,0]],
    [[-1,0,0],[0,1,0],[0,0,-1]],
    [[-1,0,0],[0,-1,0],[0,0,1]], 
    [[1,0,0],[0,-1,0],[0,0,-1]], 
    [[0,0,-1],[0,-1,0],[-1,0,0]], 
    [[0,0,1],[0,-1,0],[1,0,0]],
    [[0,0,1],[0,1,0],[-1,0,0]],
    [[0,0,-1],[0,1,0],[1,0,0]],
    [[-1,0,0],[0,0,-1],[0,-1,0]], 
    [[1,0,0],[0,0,-1],[0,1,0]], 
    [[1,0,0],[0,0,1],[0,-1,0]], 
    [[-1,0,0],[0,0,1],[0,1,0]], 
    [[0,-1,0],[-1,0,0],[0,0,-1]],
    [[0,1,0],[-1,0,0],[0,0,1]], 
    [[0,1,0],[1,0,0],[0,0,-1]],
    [[0,-1,0],[1,0,0],[0,0,1]], 
], dtype=tf.float32)

@tf.function
def moa(y_true, y_pred):
    loss = tf.reduce_mean(misorientation(y_true, y_pred))
    return loss

def misorientation(eulers_set_1, eulers_set_2):
    r1 = eulers_to_rot_mat(eulers_set_1)
    r2 = eulers_to_rot_mat(eulers_set_2)
    r2 = tf.linalg.inv(r2)
    angles = [_angle_function_acos(r1, r2, SYMMETRIES[k]) for k in range(24)]
    min_angles = tf.reduce_min(tf.abs(angles), axis=0)
    return min_angles

def _angle_function_acos(r1, r2, sym):
    traces = rot_mat_to_trace(r1, r2, sym)
    angle = tf.acos((traces-1)/2)
    return angle