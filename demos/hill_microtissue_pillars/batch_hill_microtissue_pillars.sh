#!/bin/bash
#
# Microtissue between pillars (Hill chain, nondimensionalised), full pipeline in Docker (asimov):
# stage A (biaxial, mpirun) -> cut to core (serial) -> stage B (release, mpirun) -> post-processing.
#
# Image: docker/hill_playground (built by docker/hill_playground/build.sh): fenicsoft at the ref of
# .devcontainer/fenicsoft.env, restructured dolfin_mech, hill_playground installed editable.
# Only this folder is mounted (/app): it can be a copy anywhere (outputs and logs are written here).
#
# hill_playground that runs:
#   HILL_PLAYGROUND_DIR=<playground clone>  that clone, mounted over the editable install (rsync'd changes are
#                                           used without rebuilding the image);
#   empty                                    ../.. if this folder sits in a playground clone, otherwise the
#                                           snapshot taken when the image was built.
#
# Usage (from this folder):
#   HILL_PLAYGROUND_DIR=~/documents/git/Hill-preparation-playground \
#       nohup ./batch_hill_microtissue_pillars.sh > pipeline_pillars.log 2>&1 &

set -e

REQ_CPUS=30
REQ_MEM_GB=128
SHM_SIZE=32g

JOB_STAGE_A=hill_pillars_stage_a
JOB_CUT=hill_pillars_cut_to_core
JOB_STAGE_B=hill_pillars_stage_b
JOB_NAME_PP=hill_pillars_pp

IMAGE_NAME="${IMAGE_NAME:-hill_playground:$(id -un)}"
# JIT-compiled forms kept across the four containers (and runs): the forms are compiled once.
DIJITSO_VOLUME="${DIJITSO_VOLUME:-hill-dijitso-cache-$(id -un)}"

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
HILL_IN_IMAGE=/home/fenics_user/hill_playground
HILL_PLAYGROUND_DIR="${HILL_PLAYGROUND_DIR:-}"
if [ -z "${HILL_PLAYGROUND_DIR}" ] && grep -qs 'name = "hill_playground"' "${DEMO_DIR}/../../pyproject.toml"; then
    HILL_PLAYGROUND_DIR="${DEMO_DIR}/../.."
fi
HILL_MOUNT=()
if [ -n "${HILL_PLAYGROUND_DIR}" ]; then
    HILL_PLAYGROUND_DIR="$(cd "${HILL_PLAYGROUND_DIR}" && pwd)"
    [ -d "${HILL_PLAYGROUND_DIR}/src/hill_playground" ] \
        || { echo "Error: no src/hill_playground in ${HILL_PLAYGROUND_DIR}."; exit 1; }
    HILL_MOUNT=(--mount "type=bind,src=${HILL_PLAYGROUND_DIR},dst=${HILL_IN_IMAGE},readonly")
fi

LOG_DIR=logs/pipeline_pillars
mkdir -p "${DEMO_DIR}/${LOG_DIR}"

docker_run() {
    # $1: container name, $2: cpus, $3: command run in /app (this folder)
    docker run -d --rm \
        -u "$(id -u):$(id -g)" \
        --name "$1" \
        --cpus="$2" \
        --memory="${REQ_MEM_GB}g" \
        --shm-size="${SHM_SIZE}" \
        --mount type=bind,src="${DEMO_DIR}",dst=/app \
        "${HILL_MOUNT[@]}" \
        --mount type=volume,src="${DIJITSO_VOLUME}",dst=/home/fenics_user/.cache/dolfin \
        -w /app \
        "${IMAGE_NAME}" \
        sh -c "$3"
}

run_stage() {
    local job_name="$1"
    local n_ranks="$2"
    local script="$3"

    echo "=== ${job_name}: starting (${n_ranks} rank(s), ${script}) ==="

    local cmd
    if [ "${n_ranks}" -eq 1 ]; then
        cmd="python3 ${script}"
    else
        cmd="mpirun -np ${n_ranks} python3 ${script}"
    fi

    SIM_ID=$(docker_run "${job_name}" "${n_ranks}" \
        "exec timeout 24h ${cmd} > ${LOG_DIR}/${job_name}.stdout.log 2> ${LOG_DIR}/${job_name}.stderr.log")

    echo "${job_name} running in background (ID: ${SIM_ID:0:12})..."
    EXIT_CODE=$(docker wait "${job_name}")

    if [ "${EXIT_CODE}" -ne 0 ]; then
        echo "Error: ${job_name} failed (exit code ${EXIT_CODE}). Halting pipeline."
        echo "See ${LOG_DIR}/${job_name}.stderr.log for details."
        exit 1
    fi
    echo "${job_name} finished successfully."
}

# 0. Image check + provenance (fenicsoft / dolfin_mech refs and commits) next to the logs.
docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 \
    || { echo "Error: image ${IMAGE_NAME} missing, run docker/hill_playground/build.sh first."; exit 1; }
{
    docker run --rm "${IMAGE_NAME}" cat /home/fenics_user/.fenicsoft_installed
    if [ -n "${HILL_PLAYGROUND_DIR}" ]; then
        echo "hill_playground_run=mounted ${HILL_PLAYGROUND_DIR}"
        git -C "${HILL_PLAYGROUND_DIR}" log -1 --format='hill_playground_commit=%H %s' 2>/dev/null || true
        git -C "${HILL_PLAYGROUND_DIR}" diff --quiet HEAD -- 2>/dev/null \
            || echo "hill_playground_dirty=yes (uncommitted changes or no git)"
    else
        echo "hill_playground_run=image snapshot"
    fi
} > "${DEMO_DIR}/${LOG_DIR}/image.txt"
echo "Image ${IMAGE_NAME}:"; sed 's/^/    /' "${DEMO_DIR}/${LOG_DIR}/image.txt"

# 1. Stage A -> cut -> stage B.
run_stage "${JOB_STAGE_A}" "${REQ_CPUS}" "pillars_biaxial.py"
run_stage "${JOB_CUT}" 1 "pillars_cut_to_core.py"
run_stage "${JOB_STAGE_B}" "${REQ_CPUS}" "pillars_uniaxial_release.py"

echo "=== Pipeline complete: stage A -> cut -> stage B all succeeded. ==="

# 2. Post-processing (orientation + Phi_c, p), central ROI and whole gel, both stages.
echo "Starting post-processing for stages A and B (ROI + whole gel)..."
PP_ID=$(docker_run "${JOB_NAME_PP}" 1 'timeout 2h sh -c "
        set -e &&
        python3 pillars_post_processing.py --stage a &&
        python3 pillars_post_processing.py --stage a --whole-domain &&
        python3 pillars_post_processing.py --stage b &&
        python3 pillars_post_processing.py --stage b --whole-domain
    " > '"${LOG_DIR}"'/stdout_pp.log 2> '"${LOG_DIR}"'/stderr_pp.log')

echo "Post-processing container running in background (ID: ${PP_ID:0:12})..."
PP_EXIT=$(docker wait "${JOB_NAME_PP}")

if [ "${PP_EXIT}" -ne 0 ]; then
    echo "Error: post-processing failed (exit code ${PP_EXIT})."
    echo "See ${LOG_DIR}/stderr_pp.log for details."
    exit 1
fi

echo "=== All pipeline tasks completed successfully! ==="
