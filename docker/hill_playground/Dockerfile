# syntax=docker/dockerfile:1.10
#
# Hill playground runtime image (asimov batches).
#
#   image = fenicsoft base (docker/dolfin of fenicsoft) + fenicsoft@FENICSOFT_REF + restructured dolfin_mech
#           + hill_playground, editable at /home/fenics_user/hill_playground
#
# fenicsoft and dolfin_mech: devcontainer script (.devcontainer/install_fenicsoft.sh, config fenicsoft.env), i.e.
# the same stack as VS Code. hill_playground: `pip install -e` of a snapshot of src/ taken at build time; a batch
# that mounts a playground clone over /home/fenics_user/hill_playground runs that clone instead (no rebuild).
#
# Build with docker/hill_playground/build.sh (context = playground root, filtered by Dockerfile.dockerignore).

ARG BASE_IMAGE=dolfin:v2019.2.0.dev0
FROM ${BASE_IMAGE}

# Empty = value of .devcontainer/fenicsoft.env.
ARG FENICSOFT_SOURCE=
ARG FENICSOFT_REF=
ARG DOLFIN_MECH_REF=
# New value at each build.sh call: branch heads (dolfin_mech, remote fenicsoft) are fetched again.
ARG CACHE_BUST=

COPY --chown=fenics_user:fenics .devcontainer/fenicsoft.env .devcontainer/install_fenicsoft.sh \
	/home/fenics_user/.hill_install/

# fenicsoft_src : host fenicsoft clone, read-only bind (not copied in the image; empty for FENICSOFT_SOURCE=remote).
# fenicsoft_token: GitLab token, FENICSOFT_SOURCE=remote only (never written in a layer).
RUN --mount=type=bind,from=fenicsoft_src,target=/tmp/fenicsoft_src \
	--mount=type=secret,id=fenicsoft_token,env=FENICSOFT_TOKEN \
	echo "[hill] build ${CACHE_BUST}" && \
	FENICSOFT_LOCAL_REPO=/tmp/fenicsoft_src \
	FENICSOFT_SOURCE="${FENICSOFT_SOURCE}" \
	FENICSOFT_REF="${FENICSOFT_REF}" \
	DOLFIN_MECH_REF="${DOLFIN_MECH_REF}" \
	bash /home/fenics_user/.hill_install/install_fenicsoft.sh

# hill_playground, editable: the import path points to /home/fenics_user/hill_playground/src.
COPY --chown=fenics_user:fenics pyproject.toml README.md /home/fenics_user/hill_playground/
COPY --chown=fenics_user:fenics src/ /home/fenics_user/hill_playground/src/
RUN pip install --no-cache-dir -e /home/fenics_user/hill_playground && \
	echo "hill_playground=editable /home/fenics_user/hill_playground (snapshot ${CACHE_BUST})" \
		>> /home/fenics_user/.fenicsoft_installed

# What is installed: /home/fenics_user/.fenicsoft_installed (fenicsoft source/ref/commit, dolfin_mech ref, snapshot).
CMD ["/bin/bash"]
