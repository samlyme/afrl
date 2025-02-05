# AFRL Reseach

## Git Hooks

Jupyter Notebooks don't play nice with git, since the output is stored in the actual file. In certain cases, the output is random, so the act of running the notebook can create changes. This will create unecessary merge conflicts in our repo.

To remedy this, we must clear the output of the notebooks before commits. This gaurantees that only meaningful changes to the code/markdown are captured by git. 