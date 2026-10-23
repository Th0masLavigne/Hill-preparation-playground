#!/bin/bash
#
# Microtissue between pillars (Hill chain, nondimensionalised), full pipeline in Docker (asimov):
# stage A (biaxial, mpirun) -> cut to core (serial) -> stage B (release, mpirun) -> post-processing.
#
# The whole playground is mounted (hill_playground is read from its src/ through PYTHONPATH); the image
# must hold fenicsoft (branch with scale_factor and OrientationChange) and the restructured dolfin_mech.
#
# Usage (from this folder):
#   chmod +x batch_hill_microtissue_pillars.sh
#   nohup ./batch_hill_microtissue_pillars.sh > pipeline_pillars.log 2>&1 &

set -e

REQ_CPUS=30
REQ_MEM_GB=128
SHM_SIZE=32g

JOB_STAGE_A=hill_pillars_stage_a
JOB_CUT=hill_pillars_cut_to_core
JOB_STAGE_B=hill_pillars_stage_b
JOB_NAME_PP=hill_pillars_pp

IMAGE_NAME=fenicsoft:latest

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${DEMO_DIR}/../.." && pwd)"
DEMO_REL="${DEMO_DIR#"${REPO_ROOT}"/}"
WORK_DIR="/app/${DEMO_REL}"

LOG_DIR=logs/pipeline_pillars
mkdir -p "${DEMO_DIR}/${LOG_DIR}"

docker_run() {
    # $1: container name, $2: cpus, $3: command run in WORK_DIR
    docker run -d --rm \
        -u "$(id -u):$(id -g)" \
        --name "$1" \
        --cpus="$2" \
        --memory="${REQ_MEM_GB}g" \
        --shm-size="${SHM_SIZE}" \
        --mount type=bind,src="${REPO_ROOT}",dst=/app \
        -e PYTHONPATH=/app/src \
        -w "${WORK_DIR}" \
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
