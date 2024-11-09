import os
import shutil

# Source and destination directories
src_dir = "smoothie_data/multi_model_results_old"
dst_dir = "smoothie_data/multi_model_results"

# Ensure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Walk through the source directory
for root, dirs, files in os.walk(src_dir):
    for file in files:
        # Construct the full file paths
        src_file = os.path.join(root, file)
        dst_file = os.path.join(dst_dir, os.path.relpath(src_file, src_dir))

        # Check if the file should be skipped
        skip_keywords = [
            "smoothie",
            "uniform",
            "scores",
            "pair_rm",
            "pick_random",
            "mbr",
            "oracle",
        ]
        if any(keyword in file.lower() for keyword in skip_keywords):
            # print(f"Skipping: {src_file}")
            continue

        # Remove /7b/, /1b/, /3b/ from the destination path if they exist
        dst_parts = dst_file.split(os.sep)
        if any(part in ["7b", "1b", "3b"] for part in dst_parts):
            dst_parts = [part for part in dst_parts if part not in ["7b", "1b", "3b"]]
            # Insert 'model_gens' before the filename
            dst_parts.insert(-1, "model_gens")
            dst_file = os.path.join(*dst_parts)

        # Ensure the new destination directory exists
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        # Copy the file (commented out for testing)
        shutil.copy2(src_file, dst_file)

        # Print which file is being copied (for testing)
        print(f"Copying: {src_file} to {dst_file}")


print("File copying process completed.")
