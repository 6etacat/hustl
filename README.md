# HUSTL

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