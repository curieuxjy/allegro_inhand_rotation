#!/usr/bin/env python3
# Copyright (c) 2025 Wonik Robotics
#
# This software is licensed under the MIT License.
# See the LICENSE file in the project root for full license text.

"""
Two-hands deployment script for Allegro hands (left + right)

Launches two completely independent subprocesses - one for each hand.
This is equivalent to running the single-hand scripts in two separate terminals.
"""
import os
import sys
import time
import signal
import subprocess
from pathlib import Path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-Hands Allegro Hand Deployment (Subprocess Launcher)"
    )
    parser.add_argument("--checkpoint-right", type=str, required=True,
                        help="Path to checkpoint file for right hand (.pth)")
    parser.add_argument("--checkpoint-left", type=str, default=None,
                        help="Path to checkpoint file for left hand (.pth). If not specified, uses same as right.")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Control frequency in Hz (default: 20.0)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu). Default: cuda")

    args = parser.parse_args()

    checkpoint_right = args.checkpoint_right
    checkpoint_left = args.checkpoint_left if args.checkpoint_left else args.checkpoint_right

    # Verify checkpoints exist
    if not os.path.exists(checkpoint_right):
        print(f"❌ Right hand checkpoint not found: {checkpoint_right}")
        sys.exit(1)
    if not os.path.exists(checkpoint_left):
        print(f"❌ Left hand checkpoint not found: {checkpoint_left}")
        sys.exit(1)

    run_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"🧠🧠 Starting Two-Hands deployment (subprocess) at {run_start_time}")
    print(f"📦 Right: {checkpoint_right}")
    print(f"📦 Left:  {checkpoint_left}")
    print()

    # Build commands for each hand
    # Using the single-hand scripts with --use-side-prefix for two-hands joint names
    base_dir = Path(__file__).parent

    cmd_right = [
        sys.executable, "-m", "hora.algo.deploy.deploy_ros2_right",
        "--checkpoint", checkpoint_right,
        "--hz", str(args.hz),
        "--device", args.device,
        "--use-side-prefix",  # Use ahr_* joint names
    ]

    cmd_left = [
        sys.executable, "-m", "hora.algo.deploy.deploy_ros2_left",
        "--checkpoint", checkpoint_left,
        "--hz", str(args.hz),
        "--device", args.device,
        "--use-side-prefix",  # Use ahl_* joint names
    ]

    print(f"🚀 Launching right hand process...")
    print(f"   Command: {' '.join(cmd_right)}")
    proc_right = subprocess.Popen(
        cmd_right,
        stdout=sys.stdout,
        stderr=sys.stderr,
        # Each process gets its own process group for clean signal handling
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
    )

    # Small delay to avoid ROS2 initialization conflicts
    time.sleep(1.0)

    print(f"🚀 Launching left hand process...")
    print(f"   Command: {' '.join(cmd_left)}")
    proc_left = subprocess.Popen(
        cmd_left,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
    )

    print()
    print(f"🧠🧠 Both hands running. Press Ctrl+C to stop both.")
    print()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n🧠🧠 Stopping both hands...")
        # Send SIGINT to both processes
        try:
            if proc_right.poll() is None:
                os.killpg(os.getpgid(proc_right.pid), signal.SIGINT)
        except Exception:
            proc_right.terminate()
        try:
            if proc_left.poll() is None:
                os.killpg(os.getpgid(proc_left.pid), signal.SIGINT)
        except Exception:
            proc_left.terminate()

    signal.signal(signal.SIGINT, signal_handler)

    # Wait for both processes
    try:
        while proc_right.poll() is None or proc_left.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    # Final cleanup
    proc_right.wait(timeout=5.0)
    proc_left.wait(timeout=5.0)

    run_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"🧠🧠 Two-Hands Deployment stopped.")

    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t0 = time.mktime(time.strptime(run_start_time, fmt))
        t1 = time.mktime(time.strptime(run_end_time, fmt))
        elapsed = int(round(t1 - t0))
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        print(f"🧠🧠 Total Running Time: {hrs:02d}:{mins:02d}:{secs:02d}")
    except Exception:
        print(f"🔥 Run started at {run_start_time}, ended at {run_end_time}")


if __name__ == "__main__":
    main()
