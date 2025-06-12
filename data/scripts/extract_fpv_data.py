# /// script
# requires-python = ">=3.14"
# dependencies = [
# ]
# ///

import os
import subprocess

def unzip_all(
    input_path: str = '../archives',
    output_path: str = '../raw',
):
    """
    Unzips archives and puts them in an output path
    """
    if not os.path.isdir(input_path):
        print(f"Error: Directory '{input_path}' does not exist.")
        return

    for filename in os.listdir(input_path):
        if filename.endswith(".zip"): 
            new_dir_name = os.path.splitext(filename)[0] 
            # print('new_dir_name', new_dir_name)

            cmd = ["unzip", os.path.join(input_path, filename), 
                   "-d", os.path.join(output_path, new_dir_name)]
            # print(cmd)

        try:
            # Run the wget command
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            return False

    return True

def main():
    output_path = "../raw"
    os.makedirs(output_path, exist_ok=True)

    input_path = "../archives"

    res = unzip_all(input_path, output_path)
    if res:
        print("Sucessfully unzipped everything")
    else:
        print("Something went wrong")


if __name__ == "__main__":
    main()