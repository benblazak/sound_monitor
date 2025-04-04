#! /usr/bin/env bash
set -e
cd "$(dirname "$0")"
. util.sh

if command_exists brew; then
    package_install ffmpeg portaudio
else
    package_install ffmpeg libavcodec-extra portaudio19-dev
fi
