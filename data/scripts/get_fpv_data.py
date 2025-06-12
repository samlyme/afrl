# /// script
# requires-python = ">=3.14"
# dependencies = [
# ]
# ///

import os
import subprocess
import threading

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
        

if __name__ == '__main__':

    output_path = "data/archives"
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