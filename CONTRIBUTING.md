# Contributing to Kubetorch
Please file an [issue](https://github.com/run-house/kubetorch/issues) if you encounter a bug.

If you would like to submit a bug-fix or improve an existing feature, please submit a pull request following the
process outlined below.

## Development Process
If you want to modify code, please follow the instructions for creating a Pull Request.

1. Fork the Github repository, and then clone the forked repo to local.
```
git clone git@github.com:<your-gh-username>/kubetorch.git
cd kubetorch
git remote add upstream https://github.com/run-house/kubetorch.git
```

2. Create a new branch for your development changes:
```
git checkout -b branch-name
```

3. Install Kubetorch
```
cd python_client
pip install -e .
```

4. Develop your features

5. Download and run pre-commit to automatically format your code using black and ruff.

```
pip install pre-commit
pre-commit run --files [FILES [FILES ...]]
```

6. Add, commit, and push your changes. Create a "Pull Request" on GitHub to submit the changes for review.

```
git push -u origin branch-name
```

## Testing
To run tests, install the dev dependencies:
```
cd python_client
pip install -e ".[dev]"
```

## Documentation
Docs source code is located in `python_client/kubetorch/docs/`. To build and review docs locally:

```
cd python_client
pip install -e ".[docs]"
cd kubetorch/docs/
make clean html
```

### Examples
Example code lives in [run-house/kubetorch-examples](https://github.com/run-house/kubetorch-examples), and is
rendered on the [Examples Page](https://www.run.house/examples). Please follow the process above to create
pull requests.
