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

## Repo structure

```
tmp/                  # Contains your local (temporary) data, such as vector databases
notebooks/            # Jupyter notebooks for exploration and experimentation
src/                  # Code utils that you might need to use frequently
```

Consider adding code to `src` as a temporary solution - most if not all of the re-usable utils code should be kept in `discovery_utils` repo instead.
