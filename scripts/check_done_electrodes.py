#!/usr/bin/env python3

import os
import re
from collections import defaultdict


def count_files_by_sid_elec(directory='.', pattern=None):
    """
    Count files grouped by <sid>_<elec_id> prefix.

    Args:
        directory: Directory to search (default: current directory)
        pattern: Optional regex pattern to match filenames
    """

    # Default pattern: matches <sid>_<elec_id>_<ending>
    # Assumes sid and elec_id contain word characters, numbers, or hyphens
    if pattern is None:
        pattern= r'^(.*?)(?=_(?:comp|prod))'

    file_counts = defaultdict(int)
    compiled_pattern = re.compile(pattern)

    try:
        # Get all files in directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        for filename in files:
            match = compiled_pattern.match(filename)
            if match:
                sid_elec_prefix = match.group(1)
                file_counts[sid_elec_prefix] += 1

        return dict(file_counts)

    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
        return {}
    except PermissionError:
        print(f"Permission denied accessing directory '{directory}'.")
        return {}


def main(dir):
    # Count files
    counts = count_files_by_sid_elec(dir)

    if not counts:
        print("No matching files found.")
        return

    # Display results
    print(f"File counts grouped by <sid>_<elec_id>:")
    print("-" * 40)

    # Sort by prefix for consistent output
    for prefix in sorted(counts.keys()):
        print(f"{prefix}: {counts[prefix]} files")

    print("-" * 40)
    print(f"Total groups: {len(counts)}")
    print(f"Total matching files: {sum(counts.values())}")


# Alternative function for more specific pattern matching
def count_with_custom_pattern(directory='.', sid_pattern=r'\w+', elec_pattern=r'\w+'):
    """
    Count files with custom patterns for sid and elec_id parts.

    Args:
        directory: Directory to search
        sid_pattern: Regex pattern for <sid> part
        elec_pattern: Regex pattern for <elec_id> part
    """

    pattern = f'^({sid_pattern}_{elec_pattern})_.*'
    return count_files_by_sid_elec(directory, pattern)


if __name__ == "__main__":
    dir = "/scratch/gpfs/tk6637/princeton/247-encoding/results/podcast/tk-podcast-777-gemma-scope-2b-pt-res-canonical-lag2k-25-all/tk-200ms-777-lay13-con32-reglasso-sig_coeffs"

    main(dir)


    # # Example of using custom patterns
    # print("\n" + "=" * 50)
    # print("Example with custom patterns (numeric IDs only):")
    # custom_counts = count_with_custom_pattern('.', r'\d+', r'\d+')
    #
    # if custom_counts:
    #     for prefix in sorted(custom_counts.keys()):
    #         print(f"{prefix}: {custom_counts[prefix]} files")