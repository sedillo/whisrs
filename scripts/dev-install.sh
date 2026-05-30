#!/usr/bin/env bash
# whisrs dev-install — rebuild the local checkout and restart the daemon.
#
# Maintainer convenience for the dev loop. End users should use install.sh
# (which downloads the prebuilt tarball from the latest GitHub Release).
#
# Usage:
#   ./scripts/dev-install.sh             # cargo install --path . (binaries → ~/.cargo/bin)
#   ./scripts/dev-install.sh --system    # build + sudo install to /usr/local/bin
#   ./scripts/dev-install.sh --no-restart  # skip the `whisrs restart` step

set -euo pipefail

# Some distros (e.g. Debian multiarch) install pkg-config files under a
# non-default path. Detect and export so the build can find system libraries
# (alsa, xkbcommon, etc.).
if [ -d /usr/lib/x86_64-linux-gnu/pkgconfig ] && \
   [ -z "${PKG_CONFIG_PATH:-}" ]; then
    export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig"
fi

GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "  ${GREEN}${BOLD}$1${RESET} $2"; }
warn()  { echo -e "  ${YELLOW}$1${RESET}"; }
error() { echo -e "  ${RED}$1${RESET}"; }
step()  { echo -e "\n${BOLD}[$1/$TOTAL] $2${RESET}"; }

TOTAL=2
SYSTEM_INSTALL=0
SKIP_RESTART=0

for arg in "$@"; do
    case "$arg" in
        --system)     SYSTEM_INSTALL=1 ;;
        --no-restart) SKIP_RESTART=1 ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            error "Unknown flag: $arg"
            exit 1
            ;;
    esac
done

# Resolve the repo root from the script's location so relative paths work
# regardless of where the script is invoked from.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$REPO_ROOT/Cargo.toml" ] || ! grep -q 'name = "whisrs"' "$REPO_ROOT/Cargo.toml"; then
    error "Could not find a whisrs Cargo.toml at $REPO_ROOT."
    echo "  Run this script from inside a whisrs checkout."
    exit 1
fi

echo -e "\n${BOLD}whisrs dev-install${RESET} — local rebuild\n"

cd "$REPO_ROOT"

# ── Step 1: Build and install ───────────────────────────────────────────

if [ "$SYSTEM_INSTALL" -eq 1 ]; then
    step 1 "Building release binaries..."
    cargo build --release
    info "Built:" "target/release/whisrs, target/release/whisrsd"

    info "Installing:" "system-wide → /usr/local/bin (sudo)"
    sudo install -m755 target/release/whisrs  /usr/local/bin/whisrs
    sudo install -m755 target/release/whisrsd /usr/local/bin/whisrsd
    info "Installed:" "/usr/local/bin/whisrs"
    info "Installed:" "/usr/local/bin/whisrsd"
else
    step 1 "Installing via cargo install --path ..."
    cargo install --path . --locked --force 2>&1 | tail -5

    if [ -f "$HOME/.cargo/bin/whisrs" ]; then
        info "Installed:" "$HOME/.cargo/bin/whisrs"
        info "Installed:" "$HOME/.cargo/bin/whisrsd"
    else
        error "cargo install did not produce ~/.cargo/bin/whisrs — check output above."
        exit 1
    fi
fi

# ── Step 2: Restart daemon ──────────────────────────────────────────────

if [ "$SKIP_RESTART" -eq 1 ]; then
    info "Skipping" "daemon restart (--no-restart)"
    echo -e "\n${GREEN}${BOLD}Done.${RESET}\n"
    exit 0
fi

step 2 "Restarting daemon..."

# Prefer the just-installed binary so we always restart with the version we
# just built, not whichever `whisrs` happens to be first in PATH.
if [ "$SYSTEM_INSTALL" -eq 1 ]; then
    WHISRS_BIN="/usr/local/bin/whisrs"
else
    WHISRS_BIN="$HOME/.cargo/bin/whisrs"
fi

if [ -x "$WHISRS_BIN" ]; then
    "$WHISRS_BIN" restart
else
    warn "Could not locate the freshly installed whisrs binary at $WHISRS_BIN."
    echo "  Run 'whisrs restart' manually."
fi

echo -e "\n${GREEN}${BOLD}Done.${RESET}\n"
