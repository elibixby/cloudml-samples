#!/bin/bash
set -e

#### STANDARD CMLE TEST HEADER ####
if [-z `git diff $KOKORO_GITHUB_PULL_REQUEST_COMMIT $KOKORO_GITHUB_COMMIT "$(dirname "$PWD")"`];
then
    echo "TEST IGNORED; directory not modified in pull request $KOKORO_GITHUB_PULL_REQUEST_NUMBER"
    exit 0
fi

cd ..

sudo apt-get install python-dev python-pip

pip install --upgrade -r $CMLE_REQUIREMENTS_FILE

gcloud auth activate-service-account --key-file "${KOKORO_GFILE_DIR}/${CMLE_KEYFILE}"
gcloud config set project $CMLE_PROJECT_ID
gcloud config set compute/region $CMLE_REGION

#### 

sh ./sample.sh
