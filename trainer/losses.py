import math
import numpy as np
import tensorflow as tf
from trainer.models.utils import get_num_of_joints

def compute_gradient(y_tensor):
    return np.gradient(y_tensor[1,:], y_tensor[0,:])


class GraphNetworkError(tf.keras.losses.Loss):
    
    def __init__(self, cfg):
        super(GraphNetworkError, self).__init__()
    
        self.cfg = cfg

    def call(self, y_true, y_pred):

        # verify that the labels hold the same data type
        y_true = tf.cast(y_true, y_pred.dtype)

        # compute l2-norm according to the features axis, result is in shape of[N,|V|]
        norm = tf.norm(y_true - y_pred, ord=2, axis=2)

        # reduce mean over the left two dimensions
        return tf.reduce_mean(norm)


class EndPointError(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):

        # verify that the labels hold the same data type
        y_true = tf.cast(y_true, y_pred.dtype)

        # compute l2-norm according to the colors channels
        norm = tf.norm(y_true - y_pred, ord=2, axis=3)

        # sum the norm and compute the mean average over the batch
        return tf.reduce_mean(tf.reduce_sum(norm, axis=(1, 2)))


class ForwardKinematicsError(tf.keras.losses.Loss):

    def __init__(self, cfg, arm_lengths):
        super(ForwardKinematicsError, self).__init__()

        self.cfg = cfg
        self.arm_lengths = arm_lengths
        self.fingers_num = len(self.arm_lengths)
        self.joint_idxs = self.get_joints_idxs() # indices for joints according to finger
        self.fk_error = None


    def call(self, y_true, y_pred):

        # verify that the labels hold the same data type
        y_true = tf.cast(y_true, y_pred.dtype)

        # compute loss for a tensor shaped as [N,T,q]
        if self.cfg.data.sequence:
            # compute forward kinematics for predictions and ground truth for batches of sequences
            p_ee_true = tf.transpose(tf.map_fn(fn=lambda t: self.get_fk(t), elems=tf.transpose(y_true, [1, 0, 2])), [1, 0, 2, 3])
            p_ee_pred = tf.transpose(tf.map_fn(fn=lambda t: self.get_fk(t), elems=tf.transpose(y_pred, [1, 0, 2])), [1, 0, 2, 3])

            # compute and return the error between the FKs using the euclidean norm
            self.fk_error = tf.norm(tf.subtract(p_ee_true,p_ee_pred), axis=-1) # output shape - [N,T,5]
            loss_val = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(self.fk_error, axis=-1), axis=-1), axis=-1)
        
        # compute loss for a tensor shaped as [N,q]
        else:
            # compute forward kinematics for predictions and ground truth
            p_ee_true = self.get_fk(joints=y_true)
            p_ee_pred = self.get_fk(joints=y_pred)

            # compute and return the error between the FKs using the euclidean norm
            loss_val = tf.reduce_mean(tf.reduce_sum(tf.norm(tf.subtract(p_ee_true,p_ee_pred), axis=-1), axis=-1), axis=-1)

        return loss_val

    def get_fk(self, joints):

        # compute and return forward kinematics for each one of the fingers
        ee_fingers = []
        for i in range(self.fingers_num):
            ee_fingers.append(self.compute_fk_single(joints=joints, finger_idx=i, theta1_idx=self.joint_idxs[i][0], theta2_idx=self.joint_idxs[i][1], theta3_idx=self.joint_idxs[i][2]))
            
        # p_ee_f4 = self.compute_fk_single(joints=joints, finger_idx=0, theta1_idx=0, theta2_idx=4, theta3_idx=8)
        # p_ee_f3 = self.compute_fk_single(joints=joints, finger_idx=1, theta1_idx=1, theta2_idx=5, theta3_idx=9)
        # p_ee_f2 = self.compute_fk_single(joints=joints, finger_idx=2, theta1_idx=2, theta2_idx=6, theta3_idx=10)
        # p_ee_f1 = self.compute_fk_single(joints=joints, finger_idx=3, theta1_idx=3, theta2_idx=7, theta3_idx=11)
        # p_ee_t = self.compute_fk_single(joints=joints, finger_idx=4, theta1_idx=12, theta2_idx=13, is_thumb=True)
        # return tf.concat([p_ee_f4,p_ee_f3,p_ee_f2,p_ee_f1,p_ee_t],axis=1)
        return tf.concat(ee_fingers, axis=1)

    def compute_fk_single(self, joints, finger_idx, theta1_idx, theta2_idx, theta3_idx):

        # convert joints space to DH conventions
        if finger_idx == 4: # thumb only
            theta1 = (joints[:,theta1_idx] * math.pi)
        else:
            theta1 = (joints[:,theta1_idx] * math.pi) - (math.pi / 2)
        theta2 = (joints[:,theta2_idx] * math.pi) + math.pi

        # compute forward kinematics for two joints 
        p_ee_x = self.arm_lengths[finger_idx,0] * tf.math.cos(theta1) + self.arm_lengths[finger_idx,1] * tf.math.cos(theta1 + theta2)
        p_ee_y = self.arm_lengths[finger_idx,0] * tf.math.sin(theta1) + self.arm_lengths[finger_idx,1] * tf.math.sin(theta1 + theta2)
        
        # add third joint if exists
        if theta3_idx is not None:
            theta3 = (joints[:,theta2_idx] * math.pi) + math.pi
            p_ee_x = p_ee_x + self.arm_lengths[finger_idx,2] * tf.math.cos(theta1 + theta2 + theta3)
            p_ee_y = p_ee_y + self.arm_lengths[finger_idx,2] * tf.math.sin(theta1 + theta2 + theta3)

        # compute and return forward kinematics for the end-effector
        # p_ee_x = tf.expand_dims(self.arm_lengths[finger_idx,0] * tf.math.cos(theta1) + self.arm_lengths[finger_idx,1] * tf.math.cos(theta1 + theta2), axis=1)
        # p_ee_y = tf.expand_dims(self.arm_lengths[finger_idx,0] * tf.math.sin(theta1) + self.arm_lengths[finger_idx,1] * tf.math.sin(theta1 + theta2), axis=1)
        
        # return end-effector position
        p_ee_x = tf.expand_dims(p_ee_x, axis=1)
        p_ee_y = tf.expand_dims(p_ee_y, axis=1)
        return tf.expand_dims(tf.concat([p_ee_x, p_ee_y], axis=1), axis=1)

    # get indices for joints according to finger
    def get_joints_idxs(self):

        if self.cfg.data.joints_version == '3' or self.cfg.data.joints_version == '4':
            return [[0,4,8],[1,5,9],[2,6,10],[3,7,11],[12,13,None]]
        else: # cfg.data.joints_version == '1'
            return [[0,4,None],[1,5,None],[2,6,None],[3,7,None],[8,9,None]]


class ConfigurationDynamicsError(tf.keras.losses.Loss):

    def __init__(self, cfg):
        super(ConfigurationDynamicsError, self).__init__()

        self.cfg = cfg
        self.configuration_size = get_num_of_joints(cfg=self.cfg)

    def set_time(self, time_t):
        self.time_t = time_t

    def call(self, y_true, y_pred):

        # verify that the labels and time hold the same data type
        y_pred = tf.cast(y_pred, y_true.dtype)
        self.time_t = tf.cast(self.time_t, y_true.dtype)

        # move the configuration dimension to the beginning ([N,T,q] => [N,q,T])
        y_true = tf.transpose(y_true, [0,2,1])
        y_pred = tf.transpose(y_pred, [0,2,1])
        y_shape = y_true.shape

        # repeat the time tensor to equal the configuration space in it shape ([N,T] => [N,q,T])
        self.time_t = tf.tile(tf.expand_dims(self.time_t, axis=1), [1, self.configuration_size, 1])

        # reshape all tensors (time, preds and trues) to prepare for derivation ([N,q,T] => [N*q,T])
        self.time_t = tf.reshape(self.time_t, shape=(-1, self.cfg.data.append))
        y_true = tf.reshape(y_true, shape=(-1, self.cfg.data.append))
        y_pred = tf.reshape(y_pred, shape=(-1, self.cfg.data.append))

        # compute approximation of dy/dt - based on np.gradient
        self.y_true_grad = compute_gradient_wrt_x(f=y_true, x=self.time_t)
        self.y_pred_grad = compute_gradient_wrt_x(f=y_pred, x=self.time_t)

        # compute the l2-norm between the gradient tensors 
        loss_val = tf.reduce_mean(tf.norm(tf.subtract(self.y_true_grad, self.y_pred_grad), axis=-1), axis=-1)
        
        # reshape back to [N,T,q] to use in forward dynamics
        self.y_true_grad = tf.reshape(self.y_true_grad, shape=y_shape)
        self.y_pred_grad = tf.reshape(self.y_pred_grad, shape=y_shape)
        self.y_true_grad  = tf.transpose(self.y_true_grad, [0,2,1])
        self.y_pred_grad = tf.transpose(self.y_pred_grad, [0,2,1])

        return tf.cast(loss_val, tf.float32)


class ForwardDynamicsError(tf.keras.losses.Loss):

    def __init__(self, cfg, arm_lengths):
        super(ForwardDynamicsError, self).__init__()

        self.cfg = cfg
        self.configuration_size = get_num_of_joints(cfg=self.cfg)
        self.arm_lengths = arm_lengths
        self.fingers_num = len(self.arm_lengths)
        self.joint_idxs = self.get_joints_idxs() # indices for joints according to finger

    def set_time_and_grads(self, time_t, y_true_grad, y_pred_grad):
        self.time_t = time_t
        self.y_true_grad = y_true_grad
        self.y_pred_grad = y_pred_grad

    def call(self, y_true, y_pred):

        # TODO: check if need to add/remove pi/2 to angles according to DH

        # verify that the labels and time hold the same data type
        y_pred = tf.cast(y_pred, y_true.dtype)
        self.time_t = tf.cast(self.time_t, y_true.dtype)

        # compute geometric jacobian J(q) for gt and predicted joints
        jacobians_true, jacobians_pred = self.get_geometric_jacobians(y_true, y_pred)

        # compute dot(p) = J(q)dot(q) for both labels and predictions
        fd_true = self.compute_forward_dynamics(y_grad_input=self.y_true_grad, jacobians_input=jacobians_true)
        fd_pred = self.compute_forward_dynamics(y_grad_input=self.y_pred_grad, jacobians_input=jacobians_pred)

        # compute loss as an euclidean distance between the two end-point velocities
        loss_val = tf.reduce_mean(tf.reduce_sum(tf.norm(tf.subtract(fd_true, fd_pred), axis=-1),axis=-1))
        return tf.cast(loss_val, tf.float32)

    def compute_forward_dynamics(self, y_grad_input, jacobians_input):

        # get joint derivatives per finger and reshape it to [N,T,num_fingers,3,1]
        y_grad_set = [tf.expand_dims(tf.gather(math.pi * y_grad_input, indices=x, axis=-1), axis=2) for x in self.joint_idxs]
        y_grad_set[-1] = tf.pad(y_grad_set[-1], paddings=tf.constant([[0,0],[0,0],[0,0],[0,1]]))
        y_grad = tf.expand_dims(tf.concat(y_grad_set, axis=2), axis=-1)

        # reshape both joint derivatives and jacobians to have shapes of [N*T*num_fingers,3,1] and [N*T*num_fingers,2,3] respectively
        target_shape = y_grad[:,:,:,:-1,:].shape
        y_grad = tf.reshape(y_grad, shape=[-1,3,1])
        jacobians = tf.reshape(jacobians_input, shape=[-1,2,3])

        # compute dot(p) = J(q)dot(q) and return result as a shape of [N,T,num_fingers,2]
        forward_dynamics = tf.matmul(jacobians, y_grad)
        return tf.squeeze(tf.reshape(forward_dynamics, shape=target_shape), axis=-1)

    def get_geometric_jacobians(self, y_true, y_pred):

        # compute geometric jacobians for both ground truth and predicted joints
        jacobians_true = self.get_geometric_jacobian_per_joints(joints_vals=y_true)
        jacobians_pred = self.get_geometric_jacobian_per_joints(joints_vals=y_pred)
        
        return jacobians_true, jacobians_pred

    def get_geometric_jacobian_per_joints(self, joints_vals):

        # TODO: notice denormalization of joint angles

        # iterate over the first four fingers
        jacobian_matrices = []
        for i, joint_idx_set in enumerate(self.joint_idxs[:-1]):

            # extract DOFs and links related to current finger
            joints = tf.gather(joints_vals, indices=joint_idx_set, axis=-1)
            links = self.arm_lengths[i]

            # compute the three components the jacobian is composed of
            first_comp = tf.concat([-links[0] * tf.math.sin((math.pi * joints[:,:,:1]) - (math.pi / 2)),
                                     links[0] * tf.math.cos((math.pi * joints[:,:,:1]) - (math.pi / 2))], axis=-1)
            second_comp = tf.concat([-links[1] * tf.math.sin(tf.reduce_sum((math.pi * joints[:,:,:-1]) + (math.pi / 2), axis=-1, keepdims=True)),
                                      links[1] * tf.math.cos(tf.reduce_sum((math.pi * joints[:,:,:-1]) + (math.pi / 2), axis=-1, keepdims=True))], axis=-1)
            third_comp = tf.concat([-links[2] * tf.math.sin(tf.reduce_sum((math.pi * joints) + (3 * math.pi / 2), axis=-1, keepdims=True)),
                                     links[2] * tf.math.cos(tf.reduce_sum((math.pi * joints) + (3 * math.pi / 2), axis=-1, keepdims=True))], axis=-1)

            # tile and pad them to have the shape of [N,T,J.rows,J.columns]
            first_comp = tf.pad(tf.expand_dims(first_comp, axis=-1), paddings=tf.constant([[0,0],[0,0],[0,0],[0,2]]))
            second_comp = tf.pad(tf.tile(tf.expand_dims(second_comp, axis=-1), [1,1,1,2]), paddings=tf.constant([[0,0],[0,0],[0,0],[0,1]]))
            third_comp = tf.tile(tf.expand_dims(third_comp, axis=-1), [1,1,1,3])
            
            # compute and add the jacobian matrix
            jacobian_matrices.append(tf.expand_dims(first_comp + second_comp + third_comp, axis=2))

        # repeat the same process for the thumb witout a third joint
        joints = tf.gather(joints_vals, indices=self.joint_idxs[-1], axis=-1)
        links = self.arm_lengths[i]

        # compute the three components the jacobian is composed of
        first_comp = tf.concat([-links[0] * tf.math.sin(math.pi * joints[:,:,:1]),
                                 links[0] * tf.math.cos(math.pi * joints[:,:,:1])], axis=-1)
        second_comp = tf.concat([-links[1] * tf.math.sin(tf.reduce_sum((math.pi * joints) + math.pi, axis=-1, keepdims=True)),
                                  links[1] * tf.math.cos(tf.reduce_sum((math.pi * joints) + math.pi, axis=-1, keepdims=True))], axis=-1)

        # tile and pad them to have the shape of [N,T,J.rows,J.columns]
        first_comp = tf.pad(tf.expand_dims(first_comp, axis=-1), paddings=tf.constant([[0,0],[0,0],[0,0],[0,2]]))
        second_comp = tf.pad(tf.tile(tf.expand_dims(second_comp, axis=-1), [1,1,1,2]), paddings=tf.constant([[0,0],[0,0],[0,0],[0,1]]))

        # compute and add the jacobian matrix
        jacobian_matrices.append(tf.expand_dims(first_comp + second_comp, axis=2))

        return tf.concat(jacobian_matrices, axis=2)

    # get indices for joints according to finger
    def get_joints_idxs(self):

        if self.cfg.data.joints_version == '3' or self.cfg.data.joints_version == '4':
            return [[0,4,8],[1,5,9],[2,6,10],[3,7,11],[12,13]]
        else: # cfg.data.joints_version == '1'
            return [[0,4],[1,5],[2,6],[3,7],[8,9]]

def compute_gradient_wrt_x(f, x):

    ax_dx = x[:,1:] - x[:,:-1]

    dx1 = ax_dx[:,0:-1]
    dx2 = ax_dx[:,1:]
    a = -(dx2)/(dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))

    # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
    f_grad_start = (f[:,1:2] - f[:,0:1]) / ax_dx[:,0:1]

    # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
    f_grad_mid = a*f[:,:-2] + b*f[:,1:-1] + c*f[:,2:]

    # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    f_grad_end = (f[:,-1:] - f[:,-2:-1]) / ax_dx[:,-1:]

    # concat all values to return values
    f_grad = tf.concat([f_grad_start, f_grad_mid, f_grad_end], axis=-1)

    return f_grad


class DeviceActionsError(tf.keras.losses.Loss):
    
    def __init__(self, cfg):
        super(DeviceActionsError, self).__init__()
    
        self.cfg = cfg

    def call(self, y_true, y_pred):

        # verify that the labels hold the same data type
        #y_true = tf.cast(y_true, y_pred.dtype)

        # reshape predictions and labels from [N,T,C] to [N*T,C]
        y_pred = tf.reshape(y_pred, shape=(-1, self.cfg.data.dev_classes))
        y_true = tf.reshape(y_true, shape=(-1, self.cfg.data.dev_classes))

        # compute cross-entropy loss between predictions and labels
        ce_val = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

        # reduce mean over the left two dimensions
        return tf.reduce_mean(ce_val)

class KLDivergence(tf.keras.losses.Loss):
    
    def __init__(self, cfg):
        super(KLDivergence, self).__init__()
    
        self.cfg = cfg

    def call(self, y_true, y_pred):

        # verify that the labels hold the same data type
        #y_true = tf.cast(y_true, y_pred.dtype)

        # rename z vector characteristics
        means = y_true
        stddev = y_pred

        # copute KL loss: D_KL(Q(z|X,y) || P(z|X))
        kl_val = -0.5*tf.reduce_mean(tf.reduce_sum((1+stddev-tf.math.pow(means, 2)-tf.math.exp(stddev)), axis=1))

        # reduce mean over the left two dimensions
        return kl_val