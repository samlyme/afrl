import os
import subprocess

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

def main():
    clone_repo()
    delete_misc()

if __name__ == "__main__":
    main()