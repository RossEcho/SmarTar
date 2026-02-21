import argparse
import os
import sys
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run project tests from cmd/powershell")
    parser.add_argument("--start", default="tests", help="Start directory for test discovery")
    parser.add_argument("--pattern", default="test*.py", help="Filename pattern for tests")
    parser.add_argument("--top", default=".", help="Top-level project directory")
    parser.add_argument("--quiet", action="store_true", help="Lower output verbosity")
    args = parser.parse_args()

    top_dir = os.path.abspath(args.top)
    start_dir = os.path.abspath(args.start)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=start_dir, pattern=args.pattern, top_level_dir=top_dir)

    runner = unittest.TextTestRunner(verbosity=1 if args.quiet else 2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
