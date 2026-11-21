# FasterCap (LGPL-2.1, https://github.com/ediloren/FasterCap), headless console build.
# Used only as an external reference solver for benchmarks: it is run as a
# separate process and never linked into or shipped with faradice.
FROM ubuntu:24.04 AS build
ARG FASTERCAP_COMMIT=b42179a8fdd25ab42fe45527282b4a738d7e7f87
ARG LINALGEBRA_COMMIT=627132d70bfd7eadd727f930286938a5a01d9914
ARG GEOMETRY_COMMIT=de03ffebfd5013b96102bd60f71c8fe8b73870e2
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates curl libwxgtk3.2-dev \
    && rm -rf /var/lib/apt/lists/*
# FasterCap expects its LinAlgebra and Geometry repositories as siblings.
WORKDIR /src
RUN for r in FasterCap:${FASTERCAP_COMMIT} LinAlgebra:${LINALGEBRA_COMMIT} Geometry:${GEOMETRY_COMMIT}; do \
        name=${r%%:*}; sha=${r#*:}; mkdir -p $name; \
        curl -fsSL https://github.com/ediloren/$name/archive/$sha.tar.gz | tar xz --strip-components=1 -C $name; \
    done
WORKDIR /build
# The LinAlgebra/Geometry static libraries need the wx flags too, and wx must
# come after them on the link line.
RUN CXXFLAGS="$(wx-config --cxxflags base)" cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFASTFIELDSOLVERS_HEADLESS=ON \
        -DCMAKE_CXX_STANDARD_LIBRARIES="$(wx-config --libs base)" /src/FasterCap \
    && make -j"$(nproc)"

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libwxbase3.2-1t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/FasterCap /usr/local/bin/FasterCap
COPY --from=build /src/FasterCap/LICENCE.txt /usr/share/doc/fastercap/LICENCE.txt
WORKDIR /work
ENTRYPOINT ["FasterCap"]
