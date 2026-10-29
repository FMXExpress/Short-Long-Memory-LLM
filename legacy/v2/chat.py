#!/usr/bin/env python3
"""
CRAG Chat Launcher

This script provides a unified entry point for both TUI and CLI interfaces.
Usage:
    python chat.py          # Launch TUI interface (default)
    python chat.py --cli     # Launch CLI interface
    python chat.py --tui     # Launch TUI interface (explicit)
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="CRAG Chat System")
    parser.add_argument('--cli', action='store_true', help='Use command-line interface')
    parser.add_argument('--tui', action='store_true', help='Use terminal UI interface (default)')
    parser.add_argument('log_file', nargs='?', help='Log file name (CLI mode only)')
    
    args = parser.parse_args()
    
    # Default to TUI unless CLI is explicitly requested
    use_cli = args.cli and not args.tui
    
    if use_cli:
        print("Starting CRAG Chat CLI...")
        # Import and run CLI version
        import run
        sys.argv = ['run.py'] + ([args.log_file] if args.log_file else [])
        run.main()
    else:
        print("Starting CRAG Chat TUI...")
        try:
            # Import and run TUI version
            import tui_chat
            tui_chat.main()
        except ImportError as e:
            if 'textual' in str(e):
                print("\nError: Textual library not installed.")
                print("Please install it with: pip install textual>=0.41.0")
                print("Or run: pip install -r requirements.txt")
                print("\nFalling back to CLI interface...")
                import run
                run.main()
            else:
                raise


if __name__ == "__main__":
    main()