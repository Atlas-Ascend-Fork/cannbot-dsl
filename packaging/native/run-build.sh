#!/usr/bin/env bash
set -eo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}
# Listing the catalog does not need CANN or framework wheels.
for arg in "$@"; do
    if [[ "$arg" == "--list" || "$arg" == "--help" || "$arg" == "-h" ]]; then
        exec "$python_bin" "$project_dir/build.py" "$@"
    fi
done

cann_env=${CANN_ENV:-}
wheel=${CANNBOTDSL_WHEEL:-}

if [[ -z "$cann_env" ]]; then
    for candidate in "$HOME/Ascend/cann/set_env.sh" "$HOME/Ascend/ascend-toolkit/latest/set_env.sh"; do
        if [[ -f "$candidate" ]]; then
            cann_env=$candidate
            break
        fi
    done
fi
if [[ -z "$cann_env" || ! -f "$cann_env" ]]; then
    echo "Set CANN_ENV to the CANN environment script before building." >&2
    exit 1
fi
unset PYTHONPATH
source "$cann_env"
set -u

if [[ -z "$wheel" ]]; then
    shopt -s nullglob
    candidates=()
    if [[ -n "${CANNBOTDSL_ROOT:-}" ]]; then
        candidates=("$CANNBOTDSL_ROOT"/build/run/payload/cannbotdsl-*.whl)
    fi
    shopt -u nullglob
    if (( ${#candidates[@]} == 0 )); then
        echo "Set CANNBOTDSL_WHEEL to the CANNBot-DSL wheel used for this build." >&2
        exit 1
    fi
    wheel=${candidates[0]}
    for candidate in "${candidates[@]:1}"; do
        if [[ "$candidate" -nt "$wheel" ]]; then
            wheel=$candidate
        fi
    done
fi

if [[ ! -f "$wheel" ]]; then
    echo "CANNBotDSL wheel does not exist: $wheel" >&2
    exit 1
fi

# Cache by wheel contents: rebuilt wheels can retain the same version.
wheel_digest=$(sha256sum "$wheel" | awk '{print $1}')
framework_site="$project_dir/.build/frameworks/$wheel_digest"
if [[ ! -f "$framework_site/.installed" ]]; then
    mkdir -p "$(dirname -- "$framework_site")"
    install_dir=$(mktemp -d "$project_dir/.build/framework-install-$wheel_digest-XXXXXX")
    trap 'rm -rf -- "$install_dir"' EXIT
    "$python_bin" -m pip install --no-deps --no-index --target "$install_dir" "$wheel"
    touch "$install_dir/.installed"
    if ! mv -T "$install_dir" "$framework_site" 2>/dev/null; then
        if [[ ! -f "$framework_site/.installed" ]]; then
            echo "Failed to publish the installed CANNBotDSL wheel: $framework_site" >&2
            exit 1
        fi
        echo "Another build installed the same CANNBotDSL wheel first; using them." >&2
        rm -rf -- "$install_dir"
    fi
    trap - EXIT
fi

export PYTHONPATH="$framework_site${PYTHONPATH:+:$PYTHONPATH}"
echo "CANNBotDSL wheel: $wheel"
exec "$python_bin" "$project_dir/build.py" "$@"
