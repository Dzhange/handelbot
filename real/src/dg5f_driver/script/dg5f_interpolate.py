import rclpy
from rclpy.node import Node
from control_msgs.msg import MultiDOFCommand
import yaml
import os
import numpy as np
import time

class DG5FInterpolate(Node):
    def __init__(self):
        super().__init__('dg5f_interpolate')

        # Load config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir) 
        config_path = os.path.join(root_dir, "config/rl_bidex_env_config.yaml")
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            self.control_freq = cfg.get('control_freq', 50)
            self.policy_freq = cfg.get('policy_freq', 10)
        
        self.get_logger().info(f"Loaded Config: control_freq={self.control_freq}, policy_freq={self.policy_freq}")

        # Interpolation Logic
        self.dt = 1.0 / self.control_freq
        self.policy_dt = 1.0 / self.policy_freq
        self.alpha_step = self.dt / self.policy_dt
        
        # --- Right Hand ---
        self.rh_sub = self.create_subscription(
            MultiDOFCommand, 
            '/dg5f_right/policy_reference', 
            self.rh_callback, 
            10
        )
        self.rh_pub = self.create_publisher(
            MultiDOFCommand, 
            '/dg5f_right/rj_dg_pospid/reference', 
            10
        )
        self.rh_curr_cmd = None
        self.rh_prev_cmd = None
        self.rh_alpha = 0.0
        self.rh_cmd_msg = MultiDOFCommand() 
        self.rh_received_first = False
        self.rh_last_recv_time = 0.0

        # --- Left Hand ---
        self.lh_sub = self.create_subscription(
            MultiDOFCommand, 
            '/dg5f_left/policy_reference', 
            self.lh_callback, 
            10
        )
        self.lh_pub = self.create_publisher(
            MultiDOFCommand, 
            '/dg5f_left/lj_dg_pospid/reference', 
            10
        )
        self.lh_curr_cmd = None
        self.lh_prev_cmd = None
        self.lh_alpha = 0.0
        self.lh_cmd_msg = MultiDOFCommand()
        self.lh_received_first = False
        self.lh_last_recv_time = 0.0

        # --- Timer ---
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("Interpolation Node Started")

    def rh_callback(self, msg):
        # New target received
        self.rh_last_recv_time = time.time()
        if not self.rh_received_first:
            self.rh_prev_cmd = np.array(msg.values)
            self.rh_curr_cmd = np.array(msg.values)
            self.rh_received_first = True
            
            self.rh_cmd_msg.dof_names = msg.dof_names
        else:
            self.rh_prev_cmd = self.rh_curr_cmd
            self.rh_curr_cmd = np.array(msg.values)
        
        self.rh_alpha = 0.0 # Reset interpolation

    def lh_callback(self, msg):
        self.lh_last_recv_time = time.time()
        if not self.lh_received_first:
            self.lh_prev_cmd = np.array(msg.values)
            self.lh_curr_cmd = np.array(msg.values)
            self.lh_received_first = True
            self.lh_cmd_msg.dof_names = msg.dof_names
        else:
            self.lh_prev_cmd = self.lh_curr_cmd
            self.lh_curr_cmd = np.array(msg.values)
        
        self.lh_alpha = 0.0

    def control_loop(self):
        if self.rh_received_first and (time.time() - self.rh_last_recv_time < 10.0):
            self.rh_alpha = min(self.rh_alpha + self.alpha_step, 1.0)
            cmd = (1.0 - self.rh_alpha) * self.rh_prev_cmd + self.rh_alpha * self.rh_curr_cmd
            self.rh_cmd_msg.values = cmd.tolist()
            self.rh_pub.publish(self.rh_cmd_msg)

        if self.lh_received_first and (time.time() - self.lh_last_recv_time < 10.0):
            self.lh_alpha = min(self.lh_alpha + self.alpha_step, 1.0)
            cmd = (1.0 - self.lh_alpha) * self.lh_prev_cmd + self.lh_alpha * self.lh_curr_cmd
            self.lh_cmd_msg.values = cmd.tolist()
            self.lh_pub.publish(self.lh_cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DG5FInterpolate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
