#!/usr/bin/env python3
"""
Example Python Script Template for Skills

This template demonstrates best practices for writing Python scripts
that can be included in Custom Skills.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Example script for Custom Skill'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def validate_input(input_path: str) -> bool:
    """
    Validate that the input file exists and is readable.

    Args:
        input_path: Path to input file

    Returns:
        True if valid, False otherwise
    """
    path = Path(input_path)
    if not path.exists():
        print(f"Error: Input file '{input_path}' does not exist", file=sys.stderr)
        return False
    if not path.is_file():
        print(f"Error: '{input_path}' is not a file", file=sys.stderr)
        return False
    return True


def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the input data and return results.

    Args:
        data: Input data dictionary

    Returns:
        Processed data dictionary
    """
    # Add your processing logic here
    result = {
        'status': 'success',
        'input_keys': list(data.keys()),
        'processed': True
    }
    return result


def main():
    """Main execution function."""
    args = parse_arguments()

    # Validate input
    if not validate_input(args.input):
        sys.exit(1)

    try:
        # Read input file
        with open(args.input, 'r') as f:
            input_data = json.load(f)

        if args.verbose:
            print(f"Processing {args.input}...")

        # Process data
        result = process_data(input_data)

        # Write output
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)

        if args.verbose:
            print(f"Results written to {args.output}")
            print(f"Status: {result['status']}")

        return 0

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
