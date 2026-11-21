# Palace (Apache-2.0, https://github.com/awslabs/palace) electrostatics solver,
# plus Gmsh (GPL, Python wheel) for meshing. Both are used only as external
# reference tools for benchmarks: run as separate processes, never linked into
# or shipped with faradice.
FROM ubuntu:24.04 AS build
ARG PALACE_VERSION=v0.19.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential gfortran cmake git ca-certificates pkg-config python3 \
        libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev zlib1g-dev libunwind-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${PALACE_VERSION} https://github.com/awslabs/palace.git /src
WORKDIR /build
RUN cmake /src -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/palace \
        -DPALACE_WITH_OPENMP=ON \
    && make -j"$(nproc)"   # the superbuild installs into CMAKE_INSTALL_PREFIX

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openmpi-bin libopenmpi3t64 libopenblas0 liblapack3 libgfortran5 libgomp1 zlib1g libunwind8 \
        python3 python3-venv libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libxi6 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir gmsh==4.13.1
COPY --from=build /opt/palace /opt/palace
ENV PATH=/opt/palace/bin:/opt/venv/bin:$PATH \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
WORKDIR /work
