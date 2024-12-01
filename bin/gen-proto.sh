#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")"

pushd ../proto
python -m grpc_tools.protoc -I. --python_out=gen --grpc_python_out=gen ./*.proto

# fix imports in the generated grpc file
# i have no idea why this isn't an arg to protoc
# https://github.com/protocolbuffers/protobuf/issues/1491
sed -i '' 's/^import collector_pb2/import proto.gen.collector_pb2/' gen/collector_pb2_grpc.py
popd
