#!/usr/bin/env python3
"""
Face-Controlled Mouse System - Entry Point

A computer vision-based mouse control system using facial gestures.
Move your head to control the mouse pointer, open mouth to click.

Author: Song Tan (Ningbo University, School of Information Science and Engineering)
"""

import sys
import argparse

from src import FaceMouseApplication, Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Face-Controlled Mouse System - Control your mouse with facial gestures"
    )
    
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    
    parser.add_argument(
        "-w", "--width",
        type=int,
        default=320,
        help="Camera width resolution (default: 320)"
    )
    
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=240,
        help="Camera height resolution (default: 240)"
    )
    
    parser.add_argument(
        "-s", "--sensitivity",
        type=float,
        default=0.2,
        help="Mouse sensitivity - effective area ratio (default: 0.2)"
    )
    
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="Smoothing factor 0-1, higher = smoother (default: 0.5)"
    )
    
    parser.add_argument(
        "--dead-zone",
        type=int,
        default=20,
        help="Dead zone in pixels for small movements (default: 20)"
    )
    
    parser.add_argument(
        "--mouth-threshold",
        type=int,
        default=15,
        help="Mouth open threshold in pixels (default: 15)"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="Face-Controlled Mouse v1.0.0"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    config = Config(
        CAMERA_WIDTH=args.width,
        CAMERA_HEIGHT=args.height,
        EFFECTIVE_RATIO=args.sensitivity,
        SMOOTH_FACTOR=args.smooth,
        DEAD_ZONE=args.dead_zone,
        MOUTH_OPEN_THRESHOLD=args.mouth_threshold
    )
    
    app = FaceMouseApplication(config)
    
    try:
        app.run(camera_id=args.camera)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        app.stop()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
