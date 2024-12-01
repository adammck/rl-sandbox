#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")"

pushd ../proto
python -m grpc_tools.protoc -I. --python_out=gen --grpc_python_out=gen ./*.proto
popd
