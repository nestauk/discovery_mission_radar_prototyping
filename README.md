# Mission Radar: prototyping repo

A repo to mess around and prototype features for "Mission Radar", Discovery team's data-driven horizon scanning product.

This repo can also be used for general messing around and trying out `discovery_utils` features.

## Installation

We're using poetry for dependency management.

Run the following commands to install depedencies.

```
poetry install
poetry install --with lint
poetry install --with test
pre-commit migrate-config
poetry run pre-commit install
```

To start an environment in your terminal

```
poetry env use python3.11
poetry shell
```

To add a new package, use poetry add:

```
poetry add package-name
```

## Updating discovery_utils package

The discovery_utils package is actively updated with new features. This means that its functionality may change over time. If you need to update the discovery_utils package, follow these instructions.

It is a bit complicated, but not sure if there is a simpler way in situations where the package is an actively updated repo that is (not yet) submitted to the pip repository.

Comment out the line `discovery-utils = {git = "https://github.com/nestauk/discovery_utils.git" ...` in pyproject.toml by adding hashtag at the very beginning, and save the changes. For example:

```toml
#discovery-utils = {git = "https://github.com/nestauk/discovery_utils.git", rev = "dev"}
```

Run `poetry lock` in your terminal and then run `poetry install` (this will uninstall the old version)

Remove the hastag, so that the line in pyproject.toml now looks like this, and save the changes:

```toml
discovery-utils = {git = "https://github.com/nestauk/discovery_utils.git", rev = "dev"}
```

Run `poetry lock` and `poetry install` again (this will install the new version)

## Repo structure

```
tmp/                  # Contains your local (temporary) data, such as vector databases
notebooks/            # Jupyter notebooks for exploration and experimentation
src/                  # Code utils that you might need to use frequently
```

Consider adding code to `src` as a temporary solution - most if not all of the re-usable utils code should be kept in `discovery_utils` repo instead.
