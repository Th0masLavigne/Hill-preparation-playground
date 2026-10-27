#!/usr/bin/env bash
# Runs on the HOST (devcontainer `initializeCommand`), before the container is created.
#   1. creates the bind-mount sources (docker refuses missing ones);
#   2. builds the base image from ../fenicsoft/docker/dolfin if it does not exist yet.
#
# Also used by docker/hill_playground/build.sh (asimov) with HILL_BASE_ONLY=1 (step 2 only) and
# BASE_IMAGE / USER_ID / GROUP_ID / FENICSOFT_DIR overridden. Defaults = devcontainer (uid 1000).
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$HERE")"
FENICSOFT_DIR="${FENICSOFT_DIR:-$WORKSPACE/../fenicsoft}"
DOCS_DIR="$WORKSPACE/../../local/build_docs_hill"
IMAGE="${BASE_IMAGE:-dolfin:v2019.2.0.dev0}"
USER_ID="${USER_ID:-1000}"
GROUP_ID="${GROUP_ID:-1000}"

if [ -z "${HILL_BASE_ONLY:-}" ]; then
	mkdir -p "$DOCS_DIR"
	if [ ! -d "$FENICSOFT_DIR/.git" ]; then
		echo "[hill] WARNING: no fenicsoft clone at $FENICSOFT_DIR." >&2
		echo "[hill]          FENICSOFT_SOURCE=local|editable will fail; clone it there or use FENICSOFT_SOURCE=remote." >&2
		mkdir -p "$FENICSOFT_DIR"
	fi
fi

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
	echo "[hill] base image $IMAGE found."
	exit 0
fi

DOCKERFILE="$FENICSOFT_DIR/docker/dolfin/Dockerfile"
if [ ! -f "$DOCKERFILE" ]; then
	echo "[hill] ERROR: image $IMAGE missing and $DOCKERFILE not found to build it." >&2
	exit 1
fi

echo "[hill] building $IMAGE from fenicsoft/docker/dolfin (first time only, several minutes)..."
# Apple silicon: export DOLFIN_PLATFORM=linux/amd64 to force x86_64 (see fenicsoft README).
docker build ${DOLFIN_PLATFORM:+--platform "$DOLFIN_PLATFORM"} \
	--build-arg USER_ID="$USER_ID" --build-arg GROUP_ID="$GROUP_ID" \
	-t "$IMAGE" -f "$DOCKERFILE" "$FENICSOFT_DIR"
