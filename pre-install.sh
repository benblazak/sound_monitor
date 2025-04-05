#! /usr/bin/env bash
set -e
cd "$(dirname "$0")"

./scripts/install-sys-deps.sh
./scripts/download-yamnet.sh
