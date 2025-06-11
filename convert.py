#!/usr/bin/env python3
"""
PKL to ZIP Converter for Decision Transformer Models
Converts old pickle-based model files to the new ZIP format
"""

import os
import sys
import argparse
import pickle
import torch
import zipfile
import json
import shutil
import numpy as np
from pathlib import Path


def safe_json_convert(obj):
    """
    Safely convert any object to a JSON-serializable format
    """
    if isinstance(obj, (str, bool, type(None))):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_convert(item) for item in obj]
    else:
        # Last resort: convert to string
        return str(obj)


def convert_pkl_to_zip(pkl_path, zip_path=None, verbose=True):
    """
    Convert a PKL file to ZIP format

    Args:
        pkl_path (str): Path to the input PKL file
        zip_path (str, optional): Path for output ZIP file. If None, auto-generated
        verbose (bool): Whether to print progress messages

    Returns:
        str: Path to the created ZIP file
    """

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")

    if zip_path is None:
        # Auto-generate ZIP path
        pkl_name = Path(pkl_path).stem
        zip_path = str(Path(pkl_path).parent / f"{pkl_name}.zip")

    if verbose:
        print(f"🔄 Converting {pkl_path} -> {zip_path}")

    # Load PKL file
    try:
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        if verbose:
            print(f"✅ Loaded PKL file successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load PKL file: {e}")

    # Extract model state dict and config
    if isinstance(pkl_data, dict):
        if "model_state_dict" in pkl_data:
            model_state_dict = pkl_data["model_state_dict"]

            # Extract configuration
            config_data = {}

            # Try to get model config with safe conversion
            if "model_config" in pkl_data:
                config_data.update(safe_json_convert(pkl_data["model_config"]))

            # Add other metadata using safe conversion
            for key, value in pkl_data.items():
                if key not in ["model_state_dict", "model_config"]:
                    config_data[key] = safe_json_convert(value)

        else:
            # Assume the entire PKL is a state dict
            model_state_dict = pkl_data
            config_data = {
                "converted_from_pkl": True,
                "original_file": os.path.basename(pkl_path),
            }
    else:
        raise ValueError("Unsupported PKL format")

    # Ensure required fields are present
    if "model_class" not in config_data:
        config_data["model_class"] = "DecisionTransformer"

    # Create temporary directory
    temp_dir = "temp_conversion"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Save model state dict
        model_path = os.path.join(temp_dir, "model_state_dict.pth")
        torch.save(model_state_dict, model_path)
        if verbose:
            print(f"✅ Saved model weights to temporary file")

        # Save configuration
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        if verbose:
            print(f"✅ Saved configuration to temporary file")

        # Create ZIP file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(model_path, "model_state_dict.pth")
            zipf.write(config_path, "config.json")

        if verbose:
            print(f"✅ Created ZIP file: {zip_path}")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return zip_path


def verify_zip_file(zip_path, verbose=True):
    """
    Verify that a ZIP file contains the expected model files

    Args:
        zip_path (str): Path to the ZIP file to verify
        verbose (bool): Whether to print verification results

    Returns:
        bool: True if verification passes
    """

    if verbose:
        print(f"🔍 Verifying ZIP file: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zipf:
            files = zipf.namelist()

            required_files = ["model_state_dict.pth", "config.json"]
            missing_files = [f for f in required_files if f not in files]

            if missing_files:
                if verbose:
                    print(f"❌ Missing required files: {missing_files}")
                return False

            # Try to load config
            config_data = json.loads(zipf.read("config.json"))
            if verbose:
                print(f"✅ Config loaded successfully")
                print(f"   Model class: {config_data.get('model_class', 'Unknown')}")

                if "observation_shape" in config_data:
                    print(f"   Observation shape: {config_data['observation_shape']}")
                if "action_dim" in config_data:
                    print(f"   Action dimension: {config_data['action_dim']}")

            # Try to load model weights
            temp_dir = "temp_verify"
            os.makedirs(temp_dir, exist_ok=True)

            try:
                zipf.extract("model_state_dict.pth", temp_dir)
                model_path = os.path.join(temp_dir, "model_state_dict.pth")
                state_dict = torch.load(model_path, map_location="cpu")

                if verbose:
                    print(f"✅ Model weights loaded successfully")
                    print(f"   Parameters: {len(state_dict)} tensors")

            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        if verbose:
            print(f"✅ ZIP file verification passed")
        return True

    except Exception as e:
        if verbose:
            print(f"❌ ZIP file verification failed: {e}")
        return False


def batch_convert_directory(input_dir, output_dir=None, verify=True, verbose=True):
    """
    Convert all PKL files in a directory to ZIP format

    Args:
        input_dir (str): Directory containing PKL files
        output_dir (str, optional): Output directory for ZIP files
        verify (bool): Whether to verify each converted file
        verbose (bool): Whether to print progress messages

    Returns:
        list: List of successfully converted file paths
    """

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if output_dir is None:
        output_dir = input_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pkl_files = list(input_path.glob("*.pkl"))

    if not pkl_files:
        if verbose:
            print(f"⚠️ No PKL files found in {input_dir}")
        return []

    if verbose:
        print(f"🔄 Found {len(pkl_files)} PKL files to convert")

    converted_files = []
    failed_files = []

    for pkl_file in pkl_files:
        try:
            zip_name = pkl_file.stem + ".zip"
            zip_path = output_path / zip_name

            convert_pkl_to_zip(str(pkl_file), str(zip_path), verbose=False)

            if verify:
                if verify_zip_file(str(zip_path), verbose=False):
                    converted_files.append(str(zip_path))
                    if verbose:
                        print(f"✅ {pkl_file.name} -> {zip_name}")
                else:
                    failed_files.append(str(pkl_file))
                    if verbose:
                        print(f"❌ {pkl_file.name} -> {zip_name} (verification failed)")
            else:
                converted_files.append(str(zip_path))
                if verbose:
                    print(f"✅ {pkl_file.name} -> {zip_name}")

        except Exception as e:
            failed_files.append(str(pkl_file))
            if verbose:
                print(f"❌ {pkl_file.name} (conversion failed: {e})")

    if verbose:
        print(f"\n📊 Conversion Summary:")
        print(f"   ✅ Successfully converted: {len(converted_files)}")
        print(f"   ❌ Failed: {len(failed_files)}")

        if failed_files:
            print(f"\n❌ Failed files:")
            for f in failed_files:
                print(f"   - {f}")

    return converted_files


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Convert PKL model files to ZIP format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_pkl_to_zip.py model.pkl
  
  # Convert single file with custom output
  python convert_pkl_to_zip.py model.pkl --output new_model.zip
  
  # Convert all PKL files in a directory
  python convert_pkl_to_zip.py --directory ./models
  
  # Convert directory with custom output location
  python convert_pkl_to_zip.py --directory ./old_models --output ./new_models
  
  # Verify converted files
  python convert_pkl_to_zip.py --verify model.zip
        """,
    )

    parser.add_argument(
        "input", nargs="?", help="Input PKL file to convert or ZIP file to verify"
    )

    parser.add_argument("--output", "-o", help="Output ZIP file path or directory")

    parser.add_argument(
        "--directory", "-d", help="Convert all PKL files in this directory"
    )

    parser.add_argument(
        "--verify",
        "-v",
        action="store_true",
        help="Verify a ZIP file instead of converting",
    )

    parser.add_argument(
        "--no-verify", action="store_true", help="Skip verification of converted files"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    try:
        if args.verify:
            # Verify mode
            if not args.input:
                print("❌ Error: Input file required for verification")
                sys.exit(1)

            if verify_zip_file(args.input, verbose=verbose):
                print("✅ Verification passed")
                sys.exit(0)
            else:
                print("❌ Verification failed")
                sys.exit(1)

        elif args.directory:
            # Directory mode
            converted = batch_convert_directory(
                args.directory, args.output, verify=not args.no_verify, verbose=verbose
            )

            if converted:
                print(f"🎉 Successfully converted {len(converted)} files")
                sys.exit(0)
            else:
                print("❌ No files were converted successfully")
                sys.exit(1)

        elif args.input:
            # Single file mode
            if not args.input.endswith(".pkl"):
                print("❌ Error: Input file must be a .pkl file")
                sys.exit(1)

            zip_path = convert_pkl_to_zip(args.input, args.output, verbose=verbose)

            if not args.no_verify:
                if verify_zip_file(zip_path, verbose=verbose):
                    print("🎉 Conversion and verification successful!")
                else:
                    print("❌ Conversion completed but verification failed")
                    sys.exit(1)
            else:
                print("🎉 Conversion completed!")

        else:
            print("❌ Error: Must specify input file or directory")
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
