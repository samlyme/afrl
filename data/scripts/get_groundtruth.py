import os
import subprocess

def main():
    raw_data_path = "data/raw"
    if not os.path.isdir(raw_data_path):
        print(f"Error: Directory '{raw_data_path}' does not exist.")
        return


    clean_data_path = "data/clean"
    os.makedirs(clean_data_path, exist_ok=True)

    groundtruth = "groundtruth.txt"
    
    # only the ones with '_with_gt' have usable groundtruth data
    for filename in os.listdir(raw_data_path):
        if filename.endswith('_with_gt'):
            from_path = os.path.join(raw_data_path, filename, groundtruth)
            to_path = os.path.join(clean_data_path, filename + ".csv")
            
            cmd = ["cp", from_path, to_path]
            try:
                subprocess.run(cmd, check=True)
                print(f"Copied '{from_path}' to '{clean_data_path}' (same directory).")
            except subprocess.CalledProcessError as e:
                print(f"Failed to copy'{from_path}' to '{clean_data_path}' (same directory).")

if __name__ == "__main__":
    main()