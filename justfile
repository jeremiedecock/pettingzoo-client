# Justfile with some convenient quick ways to do common things

###############################################################################

set shell := ["bash", "-euo", "pipefail", "-c"]


## HELP #######################################################################

# Show the list of available recipes
default:
    @just --list

# Show the list of available recipes
help:
    @just --list


# RELEASE #####################################################################

# Move to a new version, in pyproject.toml and in the `pip install` URLs of the documentation and the examples
bump new:
    #!/usr/bin/env bash
    set -euo pipefail
    sed -i -E 's|^version = ".*"|version = "{{new}}"|' pyproject.toml
    sed -i -E 's|releases/download/v[0-9][^/]*/pzclient-[0-9][^-]*-py3-none-any\.whl|releases/download/v{{new}}/pzclient-{{new}}-py3-none-any.whl|' \
      README.md CSC53439EP_TUTORIAL.md examples/*.py examples/*.ipynb
    grep -n "^version" pyproject.toml
    grep -n "releases/download/v[0-9]" README.md CSC53439EP_TUTORIAL.md examples/*.py examples/*.ipynb
