### Contributing guidelines

If you want to contribute to the development of the GNN, you should create a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html) of the repository.
You can read about forking workflows [here](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow), or take a look at the contributing guidelines in the [training dataset dumper documentation](https://training-dataset-dumper.docs.cern.ch/development/#contributing-guidelines).

You should make changes inside a [feature branch](https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html) in your fork. It is generally a good idea not to work directly on the the `main` branch in your fork. Then, when your feature is ready, open a [merge request](https://docs.gitlab.com/ee/user/project/merge_requests/) to the target branch on upstream (which will usually be `main`). Once this is merged to upstream, you can `git pull upstream main` from your fork to bring back in the changes, and then fork again off `main` to start the development of a new feature. If your feature branch becomes outdated with its target, you may have to rebase or merge in the changes from the target branch, and resolve any conflicts, before you can merge.

Remember to keep you fork [up to date](https://about.gitlab.com/blog/2016/12/01/how-to-keep-your-fork-up-to-date-with-its-origin/) with upstream.

#### Code Formatting

It's good practice to document your code with module and function docstrings, and inline comments.
It's also recommended to use [black](https://github.com/psf/black) to format your contributions.
You can take a look at the [umami docs](https://umami-docs.web.cern.ch/setup/development/) for more guidelines on code style.

#### Pre-commit Checks

The `pre-commit` framework is used to validate changes.
This is installed by default, so you can run run using

```bash
pre-commit run
```
