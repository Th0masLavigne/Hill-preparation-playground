#!/usr/bin/env bash
# Runs on the HOST (devcontainer `initializeCommand`), before the container is created.
#   1. creates the bind-mount sources (docker refuses missing ones);
#   2. builds the base image from ../fenicsoft/docker/dolfin if it does not exist yet.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$HERE")"
FENICSOFT_DIR="$WORKSPACE/../fenicsoft"
DOCS_DIR="$WORKSPACE/../../local/build_docs_hill"
IMAGE="dolfin:v2019.2.0.dev0"

mkdir -p "$DOCS_DIR"

if [ ! -d "$FENICSOFT_DIR/.git" ]; then
	echo "[hill] WARNING: no fenicsoft clone at $FENICSOFT_DIR." >&2
	echo "[hill]          FENICSOFT_SOURCE=local|editable will fail; clone it there or use FENICSOFT_SOURCE=remote." >&2
	mkdir -p "$FENICSOFT_DIR"
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
	--build-arg USER_ID=1000 --build-arg GROUP_ID=1000 \
	-t "$IMAGE" -f "$DOCKERFILE" "$FENICSOFT_DIR"
