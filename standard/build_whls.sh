#!/bin/bash
set -x
echo "========="
echo "Building Wheels"
echo "========="

CURRENT_SCRIPT_DIR=$(readlink -f "$(dirname $0)")
BASE_DIR=$(readlink -f "${CURRENT_SCRIPT_DIR}/..")
SECRETS_DIR=$BASE_DIR/secrets 
WHEEL_DIR=$BASE_DIR/wheels


# build neural nets memorization wheel
cd $BASE_DIR
rm -rf dist/ # remove old wheels
poetry shell
poetry build -f wheel

# build cloud compute utils wheel
cd cloud_compute_utils
rm -rf dist/ # remove old wheels
poetry build -f wheel

# copy wheels to wheel_dir
rm -rf $WHEEL_DIR
mkdir -p $WHEEL_DIR
cp $BASE_DIR/cloud_compute_utils/dist/*.whl $WHEEL_DIR
cp $BASE_DIR/dist/*.whl $WHEEL_DIR

echo "========="
echo "Done with Wheels"
echo "========="