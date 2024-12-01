#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")"

curl -o ../proto/collector.proto https://raw.githubusercontent.com/adammck/collector/refs/heads/master/proto/collector.proto
