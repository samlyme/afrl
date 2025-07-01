# /// script
# requires-python = ">=3.14"
# dependencies = [
# ]
# ///

import os
import subprocess
import threading

import pandas as pd

def wget_download(
    url,
    output_path,
    show_progress=False
):
    """
    Download a file using wget with customizable options.

    Args:
        url (str): URL of the file to download.
        output_path (str, optional): The directory to save downloaded items.
        show_progress (bool): Display download progress bar (default: True).

    Returns:
        dict: Results including success status and message.
    """
    # Base wget command
    cmd = ['wget', url]

    if output_path:
        cmd.extend(['-P', output_path])
    if not show_progress:
        cmd.append('--quiet')

    try:
        # Run the wget command
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False

def download_thread(url: str, output_path: str):
    print(f"Start download from: {url}")
    res = wget_download(url=url, output_path=output_path, show_progress=False)
    if res:
        print(f"Successfully downloaded from: {url}")
    else:
        print(f"Download failed from: {url}")

def unzip_all(
    input_path: str = 'data/dirty/fpv-uzh/archives',
    output_path: str = 'data/dirty/fpv-uzh/raw',
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
    output_path = "data/dirty/fpv-uzh/archives"
    os.makedirs(output_path, exist_ok=True)
    urls = []

    with open('data/scripts/fpv_sources.txt', 'r') as file:
        for line in file:
            urls.append(line.strip())


    threads = []
    for url in urls:
        thread = threading.Thread(target=download_thread, args=(url, output_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()    

    # unzip files
    output_path = "data/dirty/fpv-uzh/raw"
    os.makedirs(output_path, exist_ok=True)

    input_path = "data/dirty/fpv-uzh/archives"

    res = unzip_all(input_path, output_path)
    if res:
        print("Sucessfully unzipped everything")
    else:
        print("Something went wrong")

    # only get the groundtruths
    raw_data_path = "data/dirty/fpv-uzh/raw"
    if not os.path.isdir(raw_data_path):
        print(f"Error: Directory '{raw_data_path}' does not exist.")
        return


    clean_data_path = "data/clean/fpv-uzh"
    os.makedirs(clean_data_path, exist_ok=True)

    groundtruth = "groundtruth.txt"
    
    # only the ones with '_with_gt' have usable groundtruth data
    for filename in os.listdir(raw_data_path):
        if filename.endswith('_with_gt'):
            from_path = os.path.join(raw_data_path, filename, groundtruth)
            to_path = os.path.join(clean_data_path, filename + ".csv")
            
            df = pd.read_csv(
                from_path,
                sep=' ',
                comment="#", 
                header=None,
                names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
            )

            df[["timestamp", "tx", "ty", "tz"]].to_csv(to_path)

if __name__ == "__main__":
    main()