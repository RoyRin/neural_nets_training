#!/usr/bin/env bash

set -euo pipefail

# create output dirs
mkdir -p /out/test

# run tests with pytest; test failures shouldn't cause the job to fail
echo "==> running tests with pytest..."
pytest /app/tests --junitxml=/out/test/tests.xml || RES=$?

# error 0 all tests pass, error 1 some tests fail
# error >1 bigger issue - https://docs.pytest.org/en/3.1.2/usage.html
[[ ${RES:-0} -gt 1 ]] && exit $RES

echo "==> done"

