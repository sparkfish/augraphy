# Contributing

Have a feature request? Find a bug? Want to add your augmentation to the project? Feel free to submit a [pull request](https://github.com/sparkfish/augraphy/pulls), or open an [issue](https://github.com/sparkfish/augraphy/issues) to start the discussion.

# Code Quality Infrastructure
As part of our effort to main a high quality of code in the repository, we now have some light continuous integration infrastructure in place.

Pull requests and commits to the dev branch trigger a [GitHub Action](https://github.com/sparkfish/augraphy/blob/dev/.github/workflows/main.yml) which uses `tox` to create virtualenvs for Python 3.7, 3.8, and 3.9. In each of these environments, `tox`:

1. installs project requirements
2. runs `pytest`
3. lints and reformats code with `pre-commit`

# Pre-Commit
[`pre-commit`](https://github.com/pre-commit/pre-commit) is a framework for managing pre-commit hooks, which are operations that run when you `git commit`, before the commit is added to the git log. These help us maintain a consistent code style in the repository, adhering (mostly) to the PEP8 standard and keeping to conventions used by the rest of the Python community.

## Hooks
There are [several hooks](https://github.com/sparkfish/augraphy/blob/dev/.pre-commit-config.yaml) currently enabled for the Augraphy project:

1. Fix trailing whitespace
2. Fix the end-of-file
3. Check that docstrings are defined before the code they describe
4. Check yaml files for parseable syntax
5. Check for debugger imports and breakpoint() calls
6. Check that files containing pytests have names ending in "test.py"
7. Sort entries in `requirements.txt`
8. Reorder imports in python files
9. Add a trailing comma to calls and literals
10. Run the `black` formatter
11. Run the `flake8` linter


## Using Pre-Commit
When developing for the Augraphy project, you should [install `pre-commit`](https://pre-commit.com/) and execute `pre-commit install` in your local copy of the Augraphy repository. When you finish some work and are ready to create a commit, you can either:

1. run `pre-commit run --all-files` to have `pre-commit` run the list of hooks above, or
2. `git commit` normally, which will trigger `pre-commit` to run against the staged changes

Either works fine, but in the second case you should be aware that your commit will usually **fail**, because one of the hooks will likely find an issue, and some may try to fix it. When this happens, the offending file will be modified, and you will need to stage that file again. If one of the *checks* fails, you will need to manually go and edit that file to correct the error first. I personally prefer this workflow because I get some immediate feedback from the hooks about what I did wrong.

## Output
When you run `pre-commit run --all-files` or similar, you'll see something like this:

```shell
(.venv) alex:augraphy/ (dev) $ pre-commit run --all-files             [11:07:36]
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
Check docstring is first.................................................Passed
Check Yaml...............................................................Passed
Debug Statements (Python)................................................Passed
Tests should end in _test.py.............................................Passed
Fix requirements.txt.....................................................Passed
Reorder python imports...................................................Passed
Add trailing commas......................................................Passed
black....................................................................Passed
flake8...................................................................Passed
```

# Tox
[`tox`](https://github.com/tox-dev/tox) is a tool for automating the creation of Python virtual environments and the execution of commands within them. We primarily use this for testing Augraphy with multiple versions of Python.

## Using Tox
For Augraphy development, `tox` is entirely optional as a local tool, but you should consider using it. Within the Augraphy repository, `tox` is configured to also run the `pre-commit` hooks within each virtual environment it creates.

To install it, you can run `pip install tox`.

Within your local copy of the Augraphy repository, you can then run `tox -e py` to run `tox` using the version of Python it was installed with (for me, Python 3.9.6).

The `tox -e py` invocation is needed to make `tox` *only* use the version it was installed with; if you have multiple versions of Python on your system, `tox` should be able to find those and create virtualenvs with them too. In that case, you only need to use the `tox` command, without any sub-commands or flags.

## Output
When you run `tox -e py` or similar, you'll see something like this:

```shell
(.venv) alex:augraphy/ (dev) $ tox -e py                              [11:08:13]
GLOB sdist-make: /home/alex/augraphy/setup.py
py inst-nodeps: /home/alex/augraphy/.tox/.tmp/package/1/augraphy-3.0.0.zip
py installed: attrs==21.2.0,augraphy @ file:///home/alex/augraphy/.tox/.tmp/package/1/augraphy-3.0.0.zip,backports.entry-points-selectable==1.1.0,cfgv==3.3.1,distlib==0.3.2,filelock==3.0.12,identify==2.2.13,iniconfig==1.1.1,joblib==1.0.1,nodeenv==1.6.0,numpy==1.21.2,opencv-python==4.5.3.56,packaging==21.0,Pillow==8.3.1,platformdirs==2.3.0,pluggy==1.0.0,pre-commit==2.14.1,py==1.10.0,pyparsing==2.4.7,pytest==6.2.5,PyYAML==5.4.1,scikit-learn==0.24.2,scipy==1.7.1,six==1.16.0,sklearn==0.0,threadpoolctl==2.2.0,toml==0.10.2,virtualenv==20.7.2
py run-test-pre: PYTHONHASHSEED='74040138'
py run-test: commands[0] | pytest tests
============================== test session starts ==============================
platform linux -- Python 3.9.6, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
cachedir: .tox/py/.pytest_cache
rootdir: /home/alex/augraphy
collected 1 item

tests/default_pipeline_test.py .                                          [100%]

=============================== 1 passed in 0.70s ===============================
py run-test: commands[1] | pre-commit install
pre-commit installed at .git/hooks/pre-commit
py run-test: commands[2] | pre-commit run --all-files --show-diff-on-failure
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
Check docstring is first.................................................Passed
Check Yaml...............................................................Passed
Debug Statements (Python)................................................Passed
Tests should end in _test.py.............................................Passed
Fix requirements.txt.....................................................Passed
Reorder python imports...................................................Passed
Add trailing commas......................................................Passed
black....................................................................Passed
flake8...................................................................Passed
____________________________________ summary ____________________________________
  py: commands succeeded
  congratulations :)
```
