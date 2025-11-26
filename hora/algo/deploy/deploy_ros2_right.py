#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

import time
import signal

import numpy as np
import torch

from hora.algo.deploy.robots.allegro_ros2 import start_allegro_io, stop_allegro_io

from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.utils.misc import tprint


# =========================================================
# Conversion/rearrangement utility
# =========================================================

def _action_hora2ros2_right(actions):
    """Convert hora ordering directly to ROS2 ordering for RIGHT hand

    Right hand hora order: index(0-3), thumb(4-7), middle(8-11), ring(12-15)
    ROS2 order: thumb(0-3), index(4-7), middle(8-11), ring(12-15)

    Direct mapping:
      ros2[0-3]   = hora[4-7]   (thumb)
      ros2[4-7]   = hora[0-3]   (index)
      ros2[8-11]  = hora[8-11]  (middle)
      ros2[12-15] = hora[12-15] (ring)
    """
    if isinstance(actions, torch.Tensor):
        if actions.dim() > 1:
            actions = actions.view(-1)
        cmd = actions.clone()
        cmd[[0, 1, 2, 3]] = actions[[4, 5, 6, 7]]       # thumb
        cmd[[4, 5, 6, 7]] = actions[[0, 1, 2, 3]]       # index
        cmd[[8, 9, 10, 11]] = actions[[8, 9, 10, 11]]   # middle (same)
        cmd[[12, 13, 14, 15]] = actions[[12, 13, 14, 15]]  # ring (same)
        return cmd
    else:
        a = np.asarray(actions).flatten()
        cmd = a.copy()
        cmd[[0, 1, 2, 3]] = a[[4, 5, 6, 7]]       # thumb
        cmd[[4, 5, 6, 7]] = a[[0, 1, 2, 3]]       # index
        cmd[[8, 9, 10, 11]] = a[[8, 9, 10, 11]]   # middle (same)
        cmd[[12, 13, 14, 15]] = a[[12, 13, 14, 15]]  # ring (same)
        return cmd


def _obs_ros22hora_right(o):
    """Convert ROS2 ordering directly to hora ordering for RIGHT hand

    ROS2 order: thumb(0-3), index(4-7), middle(8-11), ring(12-15)
    Right hand hora order: index(0-3), thumb(4-7), middle(8-11), ring(12-15)

    Direct mapping:
      hora[0-3]   = ros2[4-7]   (index)
      hora[4-7]   = ros2[0-3]   (thumb)
      hora[8-11]  = ros2[8-11]  (middle)
      hora[12-15] = ros2[12-15] (ring)
    """
    return np.concatenate([o[4:8], o[0:4], o[8:12], o[12:16]]).astype(np.float64)


# =========================================================
# Control agent (Timer-based)
# =========================================================

class RightHardwarePlayer:
    def __init__(self, hz: float = 20.0, device: str = "cuda"):
        torch.set_grad_enabled(False)
        self.hz = float(hz)
        self.device = device
        self.use_side_prefix = False  # Set True for two-hands setup

        # model / rms
        obs_shape = (96,)
        net_config = {
            "actions_num": 16,
            "input_shape": obs_shape,
            "actor_units": [512, 256, 128],
            "priv_mlp_units": [256, 128, 8],
            "priv_info": True,
            "proprio_adapt": True,
            "priv_info_dim": 9,
        }
        self.model = ActorCritic(net_config).to(self.device).eval()
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device).eval()
        self.sa_mean_std = RunningMeanStd((30, 32)).to(self.device).eval()

        # buffers
        self.obs_buf = torch.zeros((1, 96), dtype=torch.float32, device=self.device)
        self.proprio_hist_buf = torch.zeros((1, 30, 32), dtype=torch.float32, device=self.device)

        # limits (hora order: index, thumb, middle, ring)
        self.allegro_dof_lower = torch.tensor([
            -0.4700, -0.1960, -0.1740, -0.2270,   # Index
             0.2630, -0.1050, -0.1890, -0.1620,   # Thumb
            -0.4700, -0.1960, -0.1740, -0.2270,   # Middle
            -0.4700, -0.1960, -0.1740, -0.2270,   # Ring
        ], dtype=torch.float32, device=self.device)
        self.allegro_dof_upper = torch.tensor([
             0.4700, 1.6100, 1.7090, 1.6180,      # Index
             1.3960, 1.1630, 1.6440, 1.7190,      # Thumb
             0.4700, 1.6100, 1.7090, 1.6180,      # Middle
             0.4700, 1.6100, 1.7090, 1.6180,      # Ring
        ], dtype=torch.float32, device=self.device)

        # init_pose in ROS2 order: thumb(0-3), index(4-7), middle(8-11), ring(12-15)
        self.init_pose_ros2 = np.array([
            1.1202, 1.1374, 0.8535, -0.0852,  # Thumb
            0.0627, 1.2923, 0.3383, 0.1088,   # Index
            0.0724, 1.1983, 0.1551, 0.1499,   # Middle
            0.1343, 1.1736, 0.5355, 0.2164,   # Ring
        ], dtype=np.float64)

        # state
        self.action_scale = 1.0 / 24.0
        self.prev_target = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self.cur_target  = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self._last_obs_q = None
        self._skipped = 0
        self._last_step_t = None

        # ros
        self.timer = None
        self.allegro = None

    # ---------- utils ----------
    @staticmethod
    def _unscale(x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def _pre_physics_step(self, action):
        target = self.prev_target + self.action_scale * action
        self.cur_target = torch.clamp(target, min=self.allegro_dof_lower, max=self.allegro_dof_upper)
        self.prev_target = self.cur_target

    def _post_physics_step(self, obses):
        # 1) Normalize current observation (obses: (16,) or (1,16) tensor on self.device)
        cur_obs = self._unscale(
            obses.view(-1), self.allegro_dof_lower, self.allegro_dof_upper
        ).view(1, 16)

        # 2) Roll obs_buf (96 = 32*3)
        src64 = self.obs_buf[:, 32:96].clone()
        self.obs_buf[:, 0:64] = src64
        self.obs_buf[:, 64:80] = cur_obs
        self.obs_buf[:, 80:96] = self.cur_target

        # 3) Roll proprio_hist_buf (T=30)
        src_hist = self.proprio_hist_buf[:, 1:, :].clone()
        self.proprio_hist_buf[:, 0:-1, :] = src_hist
        self.proprio_hist_buf[:, -1, :16] = cur_obs
        self.proprio_hist_buf[:, -1, 16:32] = self.cur_target

    # ---------- timer callback ----------
    @torch.inference_mode()
    def _control_step(self):
        t0 = time.perf_counter()

        # 1) norm
        obs_norm = self.running_mean_std(self.obs_buf)

        # 2) inference
        input_dict = {
            "obs": obs_norm,
            "proprio_hist": self.sa_mean_std(self.proprio_hist_buf),
        }
        action = torch.clamp(self.model.act_inference(input_dict), -1.0, 1.0)

        # 3) update target
        self._pre_physics_step(action)

        # 4) publish command (hora -> ros2 direct)
        cmd = self.cur_target.detach().to("cpu").numpy()[0]
        ros2_cmd = _action_hora2ros2_right(cmd)
        self.allegro.command_joint_position(ros2_cmd)

        # 5) non-blocking obs update (ros2 -> hora direct)
        q_pos = self.allegro.poll_joint_position(wait=False, timeout=0.0)
        if q_pos is not None:
            hora_q = _obs_ros22hora_right(q_pos)
            obs_q = torch.from_numpy(hora_q.astype(np.float32)).to(self.device)
            self._last_obs_q = obs_q
        else:
            obs_q = self._last_obs_q
            self._skipped += 1

        if obs_q is not None:
            self._post_physics_step(obs_q)

        # 6) light jitter log
        if self._last_step_t is None:
            self._last_step_t = t0
        else:
            dt = t0 - self._last_step_t
            self._last_step_t = t0
            if int(time.time()) % 5 == 0:
                hz_est = 1.0 / max(dt, 1e-6)
                print(f"[timer] {hz_est:.2f} Hz, skipped={self._skipped}")

    # ---------- deploy ----------
    def deploy(self):
        run_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"🧠 Starting RightHardwarePlayer deployment at {run_start_time}...")

        # Start ROS2 I/O (background executor)
        self.allegro = start_allegro_io(side='right', use_side_prefix=self.use_side_prefix)

        # Warm-up (blocking) — settle hardware
        warmup = int(self.hz * 8)
        for t in range(warmup):
            tprint(f"setup {t} / {warmup}")
            self.allegro.command_joint_position(self.init_pose_ros2)
            time.sleep(1.0 / self.hz)

        # First observation (blocking once — initialization stability)
        q_pos = self.allegro.poll_joint_position(wait=True, timeout=5.0)
        if q_pos is None:
            print("❌ failed to read joint state.")
            stop_allegro_io(self.allegro)
            return

        hora_q = _obs_ros22hora_right(q_pos)
        obs_q = torch.from_numpy(hora_q.astype(np.float32)).to(self.device)
        self._last_obs_q = obs_q

        # Initialize buffers
        cur_obs_buf = self._unscale(obs_q, self.allegro_dof_lower, self.allegro_dof_upper)[None]
        self.prev_target = obs_q[None]
        for i in range(3):
            self.obs_buf[:, i*32:i*32+16] = cur_obs_buf
            self.obs_buf[:, i*32+16:i*32+32] = self.prev_target
        self.proprio_hist_buf[:, :, :16] = cur_obs_buf
        self.proprio_hist_buf[:, :, 16:32] = self.prev_target

        # Register Timer (accurate frequency)
        period = 1.0 / self.hz
        self.timer = self.allegro.create_timer(period, self._control_step)
        print(f"Deployment started (timer-based {self.hz:.1f} Hz). Ctrl+C to stop.")

        # Main thread: signal handling + keep alive
        interrupted = False

        def _sigint(_sig, _frm):
            nonlocal interrupted
            interrupted = True
        signal.signal(signal.SIGINT, _sigint)

        try:
            while not interrupted:
                time.sleep(0.2)
        finally:
            try:
                if self.timer is not None:
                    self.timer.cancel()
            except Exception:
                pass
            try:
                self.allegro.go_safe(self.init_pose_ros2)
                time.sleep(1.0)
            except Exception:
                pass
            stop_allegro_io(self.allegro)
            print("🧠 Deployment stopped cleanly.")

            run_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            fmt = "%Y-%m-%d %H:%M:%S"
            try:
                t0 = time.mktime(time.strptime(run_start_time, fmt))
                t1 = time.mktime(time.strptime(run_end_time, fmt))
                elapsed = int(round(t1 - t0))
                hrs, rem = divmod(elapsed, 3600)
                mins, secs = divmod(rem, 60)
                print(f"🧠 Total Running Time: {hrs:02d}:{mins:02d}:{secs:02d}")
            except Exception:
                print(f"🔥 Run started at {run_start_time}, ended at {run_end_time}")

    # ---------- checkpoint ----------
    def restore(self, fn):
        ckpt = torch.load(fn, map_location=self.device)
        self.running_mean_std.load_state_dict(ckpt["running_mean_std"])
        self.model.load_state_dict(ckpt["model"])
        self.sa_mean_std.load_state_dict(ckpt["sa_mean_std"])


# =========================================================
# Execution
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Right Hand Allegro Deployment")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (.pth)")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Control frequency in Hz (default: 20.0)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Auto-detects if not specified.")
    parser.add_argument("--use-side-prefix", action="store_true",
                        help="Use side-specific joint names (ahr_*) for two-hands setup")

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    agent = RightHardwarePlayer(hz=args.hz, device=device)
    agent.use_side_prefix = args.use_side_prefix  # Pass to deploy
    agent.restore(args.checkpoint)
    agent.deploy()
