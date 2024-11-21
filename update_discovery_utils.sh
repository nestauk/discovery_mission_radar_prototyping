#!/bin/bash
sed -i.bak 's/^discovery-utils/#discovery-utils/' pyproject.toml && \
poetry lock && \
poetry install && \
sed -i.bak 's/^#discovery-utils/discovery-utils/' pyproject.toml && \
poetry lock && \
poetry install && \
rm pyproject.toml.bak
