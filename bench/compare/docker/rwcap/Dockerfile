# RWCap v4 (Tsinghua University, https://numbda.cs.tsinghua.edu.cn/download/RWCap_v4_en.html),
# distributed as a binary package. Used only as an external reference solver for
# benchmarks: it is run as a separate process and never linked into or shipped
# with faradice. Only its documented command line and .cap3d input are used.
FROM ubuntu:24.04
ARG RWCAP_URL=https://numbda.cs.tsinghua.edu.cn/packages/RWCap4.tar.gz
ARG RWCAP_SHA256=006ed4f664da7aca2eb66208ed1e89df75974980de86cf5341eda63cffb6f18c
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/rwcap.tar.gz "$RWCAP_URL" \
    && echo "$RWCAP_SHA256  /tmp/rwcap.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/rwcap \
    && tar -xzf /tmp/rwcap.tar.gz -C /opt/rwcap --strip-components=1 RWCapPack/rwcap-solver4 RWCapPack/readme.1st \
    && rm /tmp/rwcap.tar.gz
# The solver looks for RWCap_tables/ in its working directory; inputs are
# passed as /work/... paths and the outputs are written next to them.
WORKDIR /opt/rwcap/rwcap-solver4
ENTRYPOINT ["/opt/rwcap/rwcap-solver4/rwcap"]
