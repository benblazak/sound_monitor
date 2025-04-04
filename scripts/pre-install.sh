#! /usr/bin/env bash
set -e
cd "$(dirname "$0")"

./install-sys-deps.sh
./download-yamnet.sh
