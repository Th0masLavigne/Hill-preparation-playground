#!/usr/bin/env bash
# Build the Hill playground runtime image (asimov), for demos/*/batch_*.sh.
#
#   docker/hill_playground/build.sh                                  # fenicsoft ref of .devcontainer/fenicsoft.env
#   FENICSOFT_REF=<branch|tag|sha> docker/hill_playground/build.sh   # one-off override
#
# Environment (all optional):
#   FENICSOFT_SOURCE  local (default, clone FENICSOFT_DIR at FENICSOFT_REF) | remote (GitLab, needs FENICSOFT_TOKEN)
#   FENICSOFT_REF, DOLFIN_MECH_REF   override .devcontainer/fenicsoft.env
#   FENICSOFT_DIR     fenicsoft clone (default: ../fenicsoft next to the playground). `git fetch` it first.
#   FENICSOFT_TOKEN   GitLab token (remote only; else read from .devcontainer/secrets.env)
#   IMAGE_NAME        default hill_playground:<user>  (the asimov docker daemon is shared: one tag per user)
#   BASE_IMAGE        default dolfin:v2019.2.0.dev0, built from fenicsoft/docker/dolfin with your uid if missing
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DEVCONTAINER="$REPO_ROOT/.devcontainer"

IMAGE_NAME="${IMAGE_NAME:-hill_playground:$(id -un)}"
BASE_IMAGE="${BASE_IMAGE:-dolfin:v2019.2.0.dev0}"
FENICSOFT_DIR="$(cd "${FENICSOFT_DIR:-$REPO_ROOT/../fenicsoft}" 2>/dev/null && pwd || echo "${FENICSOFT_DIR:-$REPO_ROOT/../fenicsoft}")"
SOURCE="${FENICSOFT_SOURCE:-$(sed -n 's/^FENICSOFT_SOURCE=//p' "$DEVCONTAINER/fenicsoft.env")}"

log() { echo "[hill-image] $*"; }
die() { echo "[hill-image] ERROR: $*" >&2; exit 1; }

# 1. Base image, with the uid/gid that will run the jobs (`docker run -u $(id -u):$(id -g)`):
#    the JIT cache (DIJITSO_CACHE_DIR) and pip user site live in /home/fenics_user.
BASE_IMAGE="$BASE_IMAGE" USER_ID="$(id -u)" GROUP_ID="$(id -g)" FENICSOFT_DIR="$FENICSOFT_DIR" HILL_BASE_ONLY=1 \
	bash "$DEVCONTAINER/initialize.sh"
base_uid="$(docker run --rm --entrypoint id "$BASE_IMAGE" -u)"
[ "$base_uid" = "$(id -u)" ] || die "$BASE_IMAGE has fenics_user uid $base_uid, you are $(id -u) (image built by someone else).
    Use your own base tag: BASE_IMAGE=dolfin:v2019.2.0.dev0-$(id -un) $0"

# 2. fenicsoft source given to the build.
tmp_ctx=""
trap 'if [ -n "$tmp_ctx" ]; then rm -rf "$tmp_ctx"; fi' EXIT
case "$SOURCE" in
	local)
		[ -d "$FENICSOFT_DIR/.git" ] || die "no fenicsoft clone at $FENICSOFT_DIR (set FENICSOFT_DIR or FENICSOFT_SOURCE=remote)."
		ctx="$FENICSOFT_DIR"
		;;
	remote)
		tmp_ctx="$(mktemp -d)"
		ctx="$tmp_ctx"
		;;
	*) die "FENICSOFT_SOURCE must be local or remote for an image (got '$SOURCE')." ;;
esac

secret_args=()
token="${FENICSOFT_TOKEN:-}"
if [ -z "$token" ] && [ -f "$DEVCONTAINER/secrets.env" ]; then
	token="$(sed -n 's/^FENICSOFT_TOKEN=//p' "$DEVCONTAINER/secrets.env")"
fi
if [ -n "$token" ]; then
	export FENICSOFT_TOKEN="$token"
	secret_args=(--secret "id=fenicsoft_token,env=FENICSOFT_TOKEN")
fi

# 3. Image.
log "building $IMAGE_NAME (base $BASE_IMAGE, fenicsoft $SOURCE @ ${FENICSOFT_REF:-<fenicsoft.env>})"
DOCKER_BUILDKIT=1 docker build \
	${DOLFIN_PLATFORM:+--platform "$DOLFIN_PLATFORM"} \
	--build-arg BASE_IMAGE="$BASE_IMAGE" \
	--build-arg FENICSOFT_SOURCE="$SOURCE" \
	--build-arg FENICSOFT_REF="${FENICSOFT_REF:-}" \
	--build-arg DOLFIN_MECH_REF="${DOLFIN_MECH_REF:-}" \
	--build-arg CACHE_BUST="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
	--build-context fenicsoft_src="$ctx" \
	"${secret_args[@]}" \
	-t "$IMAGE_NAME" -f "$HERE/Dockerfile" "$REPO_ROOT"

# 4. What is inside + smoke test: hill_playground from the image snapshot.
log "installed:"
docker run --rm "$IMAGE_NAME" cat /home/fenics_user/.fenicsoft_installed | sed 's/^/    /'
docker run --rm -u "$(id -u):$(id -g)" "$IMAGE_NAME" python3 -c "import hill_playground" \
	|| die "smoke test failed: hill_playground not importable from the image."
log "done: $IMAGE_NAME"
