# source env.sh  -- activates the venv and keeps every cache on this drive
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.venv/bin/activate"
export HF_HOME="$HERE/.cache/hf" TORCH_HOME="$HERE/.cache/torch"
export PIP_CACHE_DIR="$HERE/.cache/pip" TMPDIR="$HERE/.cache/tmp"
mkdir -p "$TMPDIR"
