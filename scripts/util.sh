#! /usr/bin/env bash

command_exists() {
    command -v "$1" &>/dev/null
}

package_install() {
    if command_exists brew; then
        for i; do
            brew list "$i" &>/dev/null || brew install "$i"
        done
    elif command_exists apt-get; then
        sudo apt-get -y install "$@"
    elif command_exists dnf; then
        sudo dnf -y install "$@"
    fi
}

config() {
    local script_dir="$(dirname "${BASH_SOURCE[0]}")"
    mise exec -- python "$script_dir/../sound_monitor/config.py" "$@"
}
