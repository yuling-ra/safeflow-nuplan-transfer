#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair corrupted NPZ file by reading only the valid arrays.
"""

import os
import numpy as np
import argparse
import zipfile
import io

def repair_npz(input_path, output_path=None):
    """Try to repair a corrupted NPZ file."""
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_repaired{ext}"
    
    print(f"Attempting to repair: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Try to read each array individually
    recovered_data = {}
    
    with zipfile.ZipFile(input_path, 'r') as z:
        print(f"\nFiles in archive: {z.namelist()}")
        
        for name in z.namelist():
            # Remove .npy extension
            array_name = name.replace('.npy', '')
            
            try:
                # Try to read this specific array
                with z.open(name) as f:
                    # Read numpy array
                    data = np.load(io.BytesIO(f.read()), allow_pickle=False)
                    recovered_data[array_name] = data
                    print(f"✓ Successfully read: {array_name} {data.shape}")
            except Exception as e:
                print(f"✗ Failed to read: {array_name}")
                print(f"  Error: {str(e)[:100]}")
    
    if not recovered_data:
        print("\n❌ No arrays could be recovered!")
        return False
    
    # Save recovered data
    print(f"\n💾 Saving {len(recovered_data)} recovered arrays to {output_path}")
    np.savez_compressed(output_path, **recovered_data)
    
    # Verify saved file
    print("\n🔍 Verifying repaired file...")
    try:
        verified = np.load(output_path)
        print(f"✓ Verification successful! Arrays: {list(verified.keys())}")
        
        # Print summary
        if 'seeds' in verified:
            print(f"\n📊 Recovered trajectories: {len(verified['seeds'])}")
            print(f"   Seed range: {verified['seeds'][0]} to {verified['seeds'][-1]}")
        
        verified.close()
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Repair corrupted NPZ file')
    parser.add_argument('input_file', type=str,
                        help='Path to corrupted NPZ file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: input_file_repaired.npz)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ File not found: {args.input_file}")
        return
    
    success = repair_npz(args.input_file, args.output)
    
    if success:
        print("\n✅ Repair completed successfully!")
    else:
        print("\n❌ Repair failed!")

if __name__ == "__main__":
    main() 