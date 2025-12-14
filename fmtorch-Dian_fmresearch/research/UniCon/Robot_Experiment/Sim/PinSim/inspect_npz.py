#!/usr/bin/env python3
"""
Script to read .npz files and output detailed information in JSON format
Including array names, shapes, data types, and statistics
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path


def inspect_npz(file_path):
    """
    Read npz file and return detailed information dictionary
    
    Args:
        file_path: Path to the npz file
        
    Returns:
        Dictionary containing file information
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        file_info = {
            "file_path": str(file_path),
            "file_size_bytes": Path(file_path).stat().st_size,
            "num_arrays": len(data.files),
            "array_names": data.files,
            "arrays": {}
        }
        
        # 遍历所有数组
        for name in data.files:
            array = data[name]
            
            array_info = {
                "name": name,
                "dtype": str(array.dtype),
                "shape": list(array.shape) if hasattr(array, 'shape') else None,
                "ndim": int(array.ndim) if hasattr(array, 'ndim') else None,
                "size": int(array.size) if hasattr(array, 'size') else None,
            }
            
            # 如果是数值类型数组，添加统计信息
            if hasattr(array, 'dtype') and np.issubdtype(array.dtype, np.number):
                try:
                    array_info["statistics"] = {
                        "min": float(np.min(array)),
                        "max": float(np.max(array)),
                        "mean": float(np.mean(array)),
                        "std": float(np.std(array)),
                        "median": float(np.median(array))
                    }
                    
                    # 检查是否有 NaN 或 Inf
                    array_info["contains_nan"] = bool(np.any(np.isnan(array)))
                    array_info["contains_inf"] = bool(np.any(np.isinf(array)))
                except Exception as e:
                    array_info["statistics_error"] = str(e)
            
            # 如果数组较小，显示前几个元素的示例
            if hasattr(array, 'size') and array.size <= 10:
                try:
                    array_info["sample_values"] = array.tolist()
                except Exception as e:
                    array_info["sample_values_error"] = str(e)
            elif hasattr(array, 'flatten'):
                try:
                    # For large arrays, show first 5 elements
                    flat = array.flatten()[:5]
                    array_info["sample_values"] = flat.tolist()
                    array_info["sample_note"] = "First 5 elements (flattened)"
                except Exception as e:
                    array_info["sample_values_error"] = str(e)
            
            # Handle special types (e.g., object arrays)
            if array.dtype == object:
                array_info["object_type"] = str(type(array.flat[0])) if array.size > 0 else "empty"
                try:
                    # Try to get information about the first object
                    if array.size > 0:
                        first_obj = array.flat[0]
                        if hasattr(first_obj, 'shape'):
                            array_info["object_shape"] = list(first_obj.shape)
                        if hasattr(first_obj, 'dtype'):
                            array_info["object_dtype"] = str(first_obj.dtype)
                except Exception as e:
                    array_info["object_inspection_error"] = str(e)
            
            file_info["arrays"][name] = array_info
        
        data.close()
        return file_info
        
    except Exception as e:
        return {
            "error": str(e),
            "file_path": str(file_path)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Read .npz file and output detailed information in JSON format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        nargs='?',
        default="RobotPin_Dataset_10k.npz",
        help="Path to the .npz file to read"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="RobotPin_Dataset_10k_analysis.json",
        help="Output JSON file path (optional, defaults to stdout)"
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation spaces (default: 2)"
    )
    parser.add_argument(
        "--ensure-ascii",
        action="store_true",
        help="Ensure ASCII encoding (default allows Unicode)"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.input_file).exists():
        print(f"Error: File '{args.input_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Read and analyze file
    print(f"Reading file: {args.input_file}", file=sys.stderr)
    file_info = inspect_npz(args.input_file)
    
    # Convert to JSON
    json_str = json.dumps(
        file_info,
        indent=args.indent,
        ensure_ascii=args.ensure_ascii
    )
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"Results saved to: {args.output}", file=sys.stderr)
    else:
        print(json_str)
    
    # Print brief summary to stderr
    if "error" not in file_info:
        print(f"\nSummary: Found {file_info['num_arrays']} arrays", file=sys.stderr)
        print(f"Array names: {', '.join(file_info['array_names'])}", file=sys.stderr)


if __name__ == "__main__":
    main()

