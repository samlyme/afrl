import argparse
import os
import subprocess
import threading

import h5py
import numpy as np
import pandas as pd


def get_fpv_uzh():
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


def get_mid_air():
    def download_and_unzip(out_dir = "data/dirty/mid-air", download_config = "data/scripts/mid-air_sources.txt"):
        # Ensure the output directory exists
        os.makedirs(out_dir, exist_ok=True)

        # Download files using wget
        wget_command = [
            "wget",
            "--content-disposition",
            "-x",
            "-nH",
            "-P",
            out_dir,
            "-i",
            download_config,
        ]
        subprocess.run(wget_command, check=True)

        # Find and unzip files
        for root, _, files in os.walk(out_dir):
            for filename in files:
                if filename.endswith(".zip"):
                    zip_filepath = os.path.join(root, filename)
                    unzip_command = ["unzip", "-o", "-d", os.path.dirname(zip_filepath), zip_filepath]
                    subprocess.run(unzip_command, check=True)

    # MID-AIR position data is sampled at 100hz
    def process_hdf5(path: str) -> list[pd.DataFrame]:
        f = h5py.File(path, "r")

        gts: list[str] = []
        def get_positions(name: str):
            if ("groundtruth/position" in name):
                gts.append(name)
                
        f.visit(get_positions)

        item  = f[gts[0]] 
        if not isinstance(item, h5py.Dataset):
            raise TypeError("Not a dataset")

        out = []
        for gt in gts:
            curr = f[gt]
            if not isinstance(curr, h5py.Dataset): continue
            df = pd.DataFrame(curr, columns=["tx", "ty", "tz"])

            # This dataset is sampled at 100hz
            interval =  1/100
            rows: int = len(df)
            timestamps = np.arange(0, rows * interval, interval)
            df.insert(0, "timestamp", timestamps)
            out.append(df)

        return out

    def walk_and_process(path: str) -> list[pd.DataFrame]:
        out = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith("hdf5"):
                    out.extend(process_hdf5(os.path.join(dirpath, filename)))

        return out


    download_and_unzip()
    
    all_pos: list[pd.DataFrame] = walk_and_process("data/dirty/mid-air")
    out_dir = "data/clean/mid-air"
    os.makedirs(out_dir, exist_ok=True)

    for index, pos in enumerate(all_pos):
        pos.to_csv(os.path.join(out_dir, f"trajectory_{index}.csv"))

def get_riotu_labs():
    # The riotu labs data is already in the correct format
    out_path = "data/clean/riotu-labs"
    def clone_repo():
        cmd = ["git", 
            "clone", 
            "https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories",
            out_path
            ]
        
        try:
            subprocess.run(cmd, check=True)
        except:
            print("Failed to clone repo")

    # deletes files like .git, README, etc.
    def delete_misc():
        cmd1 = ["rm", "-rf", os.path.join(out_path, ".git")]
        cmd2 = ["rm", os.path.join(out_path, "README.md")]
        cmd3 = ["rm", os.path.join(out_path, ".gitattributes")]

        try:
            subprocess.run(cmd1, check=True)
            subprocess.run(cmd2, check=True)
            subprocess.run(cmd3, check=True)
        except:
            print("Failed to remove misc files")

    clone_repo()
    delete_misc()

def main():
    parser = argparse.ArgumentParser(
        description="Automated data fetching process."
    )

    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        help="Specifies desired datasets. If not specified all known datasets are used."
    )

    args = parser.parse_args()
    
    datasets = { 
        "fpv-uzh": get_fpv_uzh, 
        "mid-air": get_mid_air, 
        "riotu-labs": get_riotu_labs
    }

    if not args.datasets:
        for get_func in datasets.values():
            get_func()
    else:
        for dataset in args.datasets:
            if dataset in datasets:
                datasets[dataset]()
            else:
                print(f"Unknown dataset {dataset}")
    
if __name__ == "__main__":
    main()