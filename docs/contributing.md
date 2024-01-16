### Working from a Fork

If you want to contribute to the development of Salt, you should work from a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html) of the repository.
You can read about forking workflows [here](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow), or take a look at the contributing guidelines in the [training dataset dumper documentation](https://training-dataset-dumper.docs.cern.ch/contributing/).

After creating a fork, clone the fork and add the upstream repository.
```
git clone ssh://git@gitlab.cern.ch:7999/<cern_username>/salt.git
git remote add upstream ssh://git@gitlab.cern.ch:7999/atlas-flavor-tagging-tools/algorithms/salt.git
```

??? info "Ensure your fork is not set to private"

    This prevents the results of the CI from being visible in MRs to the main repo.

You should make changes inside a [feature branch](https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html) in your fork. It is generally a good idea not to work directly on the the `main` branch in your fork. Then, when your feature is ready, open a [merge request](https://docs.gitlab.com/ee/user/project/merge_requests/) to the target branch on upstream (which will usually be `main`). Once this is merged to upstream, you can `git pull upstream main` from your fork to bring back in the changes, and then fork again off `main` to start the development of a new feature. If your feature branch becomes outdated with its target, you may have to rebase or merge in the changes from the target branch, and resolve any conflicts, before you can merge.

Remember to keep you fork [up to date](https://about.gitlab.com/blog/2016/12/01/how-to-keep-your-fork-up-to-date-with-its-origin/) with upstream.

### Contribution Workflow

When adding features or fixing bugs, it's a good idea to let people know what you plan to work on before you start working on it.
The following procedure should be followed:

1. (Optional) **Informal discussion** on [mattermost](https://mattermost.web.cern.ch/aft-algs/channels/h5-dumper) to check if the feature/bug exists, and is a suitable addition.
2. **Open an issue** on [GitLab]({{repo_url}}-/issues). This is a place to describe in more detail the feature/bug, and to work out what is necessary to change in the code.
3. **Assign a responsible person** to the issue. If the person who opened the issue has they capability, by default they should assign themselves to the issue. Otherwise the maintainers will assign a responsible person.
4. **Merge request**. The assigned person should work on the feature and open a MR. This will be reviewed, any follow-up issues created, and finally merged. The corresponding issue should be closed.

### Code Standards

Good coding standards are highly encouraged.
You can take a look at the [umami docs](https://umami-docs.web.cern.ch/setup/development/) or the coding style [tutorial](https://ftag.docs.cern.ch/software/tutorials/tutorial-coding/) for guidance on code style.
In short, aim to write clean, readible and type-hinted code with module and function docstrings, and plenty of inline comments.
Code is formatted using [black](https://github.com/psf/black) (this is enforced by the pre-commit checks).

VS Code is the recommended editor when developing for salt.
The package comes with some recommended extensions which you can review and install by running the `Extensions: Show Recommended Extensions` command.
See also the [umami guide](https://umami-docs.web.cern.ch/setup/development/VS_code/) for development with VS Code.

### Pre-commit Checks

The `pre-commit` framework is used to ensure contributions follow good coding standards.
It is installed as a package dependency.
To set it up, just run

```bash
pre-commit install
```

The pre-commit checks will then be run every time you run `git commit`.
You can run them manually over staged changes using

```bash
pre-commit run
```

Include the `--all-files` flag if you want to run the checks over all the files in the repository.


### Test Suite

From the `salt/` source directory (the one with the `tests/` dir inside), run

```bash
pytest --cov=salt --show-capture=stdout
```

Adding `--show-capture=stdout` just hides a bunch of `DEBUG` statements.

### Getting help

If you need help, you can ask on [mattermost](https://mattermost.web.cern.ch/aft-algs/channels/gnns)
or open an issue in the repository.