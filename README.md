# AFRL Reseach

## Git stuff

Jupyter Notebooks don't play nice with git, since the output is stored in the actual file. In certain cases, the output is random, so the act of running the notebook can create changes. This will create unecessary merge conflicts in our repo.

To remedy this, we must clear the output of the notebooks before commits. This gaurantees that only meaningful changes to the code/markdown are captured by git. 

### Setup

In the `.git` directory, you will find a `config` file. Add the following:
```
[filter "strip-notebook-output"]
	clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```

There should also be a `.gitattributes` file that contains the following:
```
*.ipynb filter=strip-notebook-output
```

This will "clean" the notebooks before adding to the git record, but it will leave your local file untouched. 