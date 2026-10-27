#!/usr/bin/env bash
# Install fenicsoft (+ restructured dolfin_mech) inside the devcontainer.
#
# Config: .devcontainer/fenicsoft.env (+ optional secrets.env).
# Any variable already set in the environment wins, e.g.
#     FENICSOFT_REF=64-reproduce-experiment-from-mostert-et-al-2 bash .devcontainer/install_fenicsoft.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- configuration: environment > secrets.env > fenicsoft.env -------------------------------
_src="${FENICSOFT_SOURCE:-}"
_ref="${FENICSOFT_REF:-}"
_url="${FENICSOFT_REMOTE_URL:-}"
_tok="${FENICSOFT_TOKEN:-}"
_dm="${DOLFIN_MECH_REF:-}"
set -a
# shellcheck source=/dev/null
source "$HERE/fenicsoft.env"
# shellcheck source=/dev/null
[ -f "$HERE/secrets.env" ] && source "$HERE/secrets.env"
set +a
FENICSOFT_SOURCE="${_src:-$FENICSOFT_SOURCE}"
FENICSOFT_REF="${_ref:-$FENICSOFT_REF}"
FENICSOFT_REMOTE_URL="${_url:-$FENICSOFT_REMOTE_URL}"
FENICSOFT_TOKEN="${_tok:-${FENICSOFT_TOKEN:-}}"
DOLFIN_MECH_REF="${_dm:-$DOLFIN_MECH_REF}"

LOCAL_REPO="${FENICSOFT_LOCAL_REPO:-$HOME/fenicsoft_src}"
EXTRAS="[dev,docs]"
STAMP="$HOME/.fenicsoft_installed"
DOLFIN_MECH_URL="https://github.com/th0maslavigne/dolfin_mech_devel_lavigne.git"

log() { echo "[fenicsoft] $*"; }
die() { echo "[fenicsoft] ERROR: $*" >&2; exit 1; }

require_local_repo() {
	[ -d "$LOCAL_REPO/.git" ] || die "no git clone at $LOCAL_REPO (host: ../fenicsoft next to this repo)."
	# Host files may belong to another uid (root in a docker build context): allow read-only git queries
	# on them. `git clone` of a local repo checks the .git path itself, hence both entries.
	local dir
	for dir in "$LOCAL_REPO" "$LOCAL_REPO/.git"; do
		git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$dir" \
			|| git config --global --add safe.directory "$dir"
	done
}

commit=""
case "$FENICSOFT_SOURCE" in
	local)
		require_local_repo
		for cand in "$FENICSOFT_REF" "origin/$FENICSOFT_REF"; do
			commit="$(git -C "$LOCAL_REPO" rev-parse --verify --quiet "${cand}^{commit}")" && break
		done
		[ -n "$commit" ] || die "ref '$FENICSOFT_REF' not found in $LOCAL_REPO (fetch it on the host first)."
		log "local clone, ref '$FENICSOFT_REF' -> ${commit:0:10}"
		tmp="$(mktemp -d)"
		trap 'rm -rf "$tmp"' EXIT
		# Throw-away clone sharing the objects of the host repo: nothing is written in $LOCAL_REPO.
		git clone --quiet --shared --no-checkout "$LOCAL_REPO" "$tmp/fenicsoft"
		git -C "$tmp/fenicsoft" -c advice.detachedHead=false checkout --quiet "$commit"
		pip install --no-cache-dir "$tmp/fenicsoft$EXTRAS"
		;;
	editable)
		require_local_repo
		commit="$(git -C "$LOCAL_REPO" rev-parse HEAD)"
		git -C "$LOCAL_REPO" diff --quiet HEAD -- 2>/dev/null || commit="$commit (+ uncommitted changes)"
		FENICSOFT_REF="$(git -C "$LOCAL_REPO" rev-parse --abbrev-ref HEAD)"
		log "editable install of $LOCAL_REPO (checked out: $FENICSOFT_REF)"
		pip install --no-cache-dir -e "$LOCAL_REPO$EXTRAS"
		;;
	remote)
		auth=""
		[ -n "$FENICSOFT_TOKEN" ] && auth="oauth2:${FENICSOFT_TOKEN}@"
		log "remote https://$FENICSOFT_REMOTE_URL @ $FENICSOFT_REF"
		GIT_TERMINAL_PROMPT=0 pip install --no-cache-dir \
			"fenicsoft$EXTRAS @ git+https://${auth}${FENICSOFT_REMOTE_URL}@${FENICSOFT_REF}" \
			|| die "remote install failed (token? ref? network?)."
		commit="$(python3 -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("fenicsoft").read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id", "?"))')"
		;;
	*)
		die "FENICSOFT_SOURCE must be local, editable or remote (got '$FENICSOFT_SOURCE')."
		;;
esac

log "restructured dolfin_mech @ $DOLFIN_MECH_REF"
pip install --no-cache-dir --force-reinstall --no-deps "git+${DOLFIN_MECH_URL}@${DOLFIN_MECH_REF}"

version="$(python3 -c 'import importlib.metadata as m; print(m.version("fenicsoft"))')"
cat > "$STAMP" <<STAMP_EOF
source=$FENICSOFT_SOURCE
ref=$FENICSOFT_REF
commit=$commit
version=$version
dolfin_mech_ref=$DOLFIN_MECH_REF
installed=$(date -u +%Y-%m-%dT%H:%M:%SZ)
STAMP_EOF
log "done:"
sed 's/^/    /' "$STAMP"
