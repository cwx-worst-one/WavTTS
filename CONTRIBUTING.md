# Contributing Guide

To develop new models or migrate existing model training processes, you are suggested to create a project under `recipes`,
and then implement several modules. Taking `sample_project` as an example:

- `modules/model.py`, which defines a model based on CNN;
- `modules/pl_datamodule.py`, which implements `pl.LightningDataModule` and specifies the dataset and dataloading logic;
- `modules/pl_module.py`, which implements `pl.LightningModule` and configures the optimizer and scheduler used for
training, and describes the logic for training/validation/testing;
- `conf/default.yaml`, which specifies the training parameters and defines the required objects using `HyperPyYAML`;

## Opening a Merge Request

Before opening a merge request to contribute a recipe, please run the following commands to format your project:
```bash
make style
make quality
```
Or copy [hooks/pre-commit](hooks/pre-commit) into your local folder **.git/hooks** as mentioned above.
It will automatically run commands above for you when committing.

If you are on a personal dev branch for a long time, please read `CHANGELOG.md` to see if there are any changes in
`samantha` package that may affect your project. If so, please rebase your branch onto the latest `master` branch to
avoid any merging conflicts.

## Making change to `samantha` package

If you are making changes to the `samantha` package, please record your changes in the `CHANGELOG` folder if needed.
The style and format of the changelog can be found in the `CHANGELOG.md`, and a template can be found in `CHANGELOG/template.md`.

If you want to change the functionality of the `samantha` package while you are developing a recipe,
you should first open a merge request to merge the change to `samantha` package into `master`, and then rebase your
recipe branch onto the latest `master` branch. Otherwise, you may encounter merging conflicts when you try to merge
your recipe branch into `master` in the future.

### Document

If you developed a new feature to `samantha`, please write document/doc-string to explain what it is and how to use it.

After development, you can build doc offline to see if it's right documented.

```shell
bash build_docs.sh
open docs/build/html/index.html
```

## Branch Names
We encourage developers to use the following convention when naming their
branches: `<author>/<branch-type>/<branch-name>`. For example, a valid branch
name is `john_doe/feat/add_foobar`. It is suggested to use a `branch-type` found
in `[build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test]`.

## Commit Messages
Commit messages must adhere to [these rules](https://www.conventionalcommits.org/en/v1.0.0/#specification).

## Large File
If your commits contain large file such as `*.wav`, `*.ckpt`, `*.bin`, etc.,
please run `git lfs install` before adding files in case of repo size growth.

## Merge Updates
We recommend using **rebase** or **cherry-pick** rather than **merge** to get updates from other branches. Because **merge**
will mess up our git commit history.

Here are some typical commands using **rebase** and **cherry-pick**

### rebase
If you want take all updates from other branch, you may need this op to do this.
```shell
# base on local branch
git rebase <base-branch-name> # this command will put all your current branch commit history above <base-branch-name>
git rebase master

# base on remote branch
git pull --rebase origin <base-branch-name>
git pull --rebase origin master
```
For more usage, please refer to [git rebase](https://git-scm.com/docs/git-rebase).

### cherry-pick
If you only want to take several commits from other branch, `cherry-pick` is all you need.

```shell
# single commit
git cherry-pick <commit-hash> # this will take the single commit to your current branch

# multiple commits
git cherry-pick <commit-hash-1>^..<commit-hash-n> # take all commits between [<commit-hash-1>, <commit-hash-n>], including commit-1 and commit-n

git cherry-pick <commit-hash-1>..<commit-hash-n> # take all commits between (<commit-hash-1>, <commit-hash-n>], excluding commit-1.
```
If you want to `cherry-pick` latest commit from remote, please fetch first: `git fetch --all`.

## Squash Commits

Before you submit final version to repo, please squash your commits and clean up commit messages.

```shell
1. git rebase -i <commit-before-your-first-commit>
2. pick the first one and change pick to squash to other commits, save and quit
3. clean up commit messages, save and quit.
4. git push -f <local-branch-name>:<remote-branch-name>
```

For more detail, please refer to [squash last X commits](https://www.baeldung.com/ops/git-squash-commits#1-squash-the-last-x-commits)
