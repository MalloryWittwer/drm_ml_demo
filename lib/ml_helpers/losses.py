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
def custom_loss(y_true, y_pred):
    rot_mat1 = eulers_to_rot_mat(y_true)
    rot_mat2 = eulers_to_rot_mat(y_pred)
    rot_mat_inv2 = tf.linalg.inv(rot_mat2)
    angles = [_angle_function(rot_mat1, rot_mat_inv2, SYMMETRIES[k]) for k in range(24)]
    moa_loss = tf.reduce_min(tf.abs(angles), axis=0)
    moa_loss_metric = tf.reduce_mean(moa_loss)
    return moa_loss_metric

def _angle_function(r1, r2, sym): 
    traces = rot_mat_to_trace(r1, r2, sym)
    angle = tf.sqrt(3-traces)
    return angle
