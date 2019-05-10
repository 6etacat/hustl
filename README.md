# HUSTL

## Pipeline Stages and Scripts

#### Stage 1 Optimal Frames Selection
`stabilize.py`

#### Stage 2 Color Consistency Across Frames
`exp_comp.py` <br>
`exp_comp_2.py` <br>
`exp_comp_3.py`

#### Stage 3 Video Stabilization
`frame_selection.py`

## Environment

#### Installation

`conda env create -f environment_ARCH.yml`

#### Update

`conda env update -f environment_ARCH.yml`

#### Activate/Deactivate

`conda activate hustl`

`conda deactivate`

## Github helper scripts

#### Merge `master` to local branch

Make sure you are in your branch with `git checkouth BRANCH`

`sh github_helper/fast_forward.sh`

#### Create pull request after commit and push

With custom title and comment:

`sh github_helper/pull_request.sh "TITLE" ["COMMENT"]`

Without custom title and comment:

`sh github_helper/pull_request.sh --no-edit`

## Docs and Docs Generation

Please write [numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) in your code. Refer to `utils.extract_sift_features` for example.

#### Linux or Mac OS

- Create or update docs: `make`
- Delete docs: `make clean`

#### Windows

- Create or update docs: `make.bat`
- Delete docs: `make.bat clean`
