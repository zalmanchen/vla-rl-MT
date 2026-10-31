ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS runtime-base
ARG APT_MIRROR=http://mirrors.aliyun.com/ubuntu/
ARG INSTALL_APT=1
ENV DEBIAN_FRONTEND=noninteractive
RUN if [ "$INSTALL_APT" = 1 ]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu/|${APT_MIRROR}|g; s|http://security.ubuntu.com/ubuntu/|${APT_MIRROR}|g" /etc/apt/sources.list && \
      apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl python3 ffmpeg libegl1 libgl1 libglfw3 libglib2.0-0 \
        libvulkan1 vulkan-tools libgomp1 libsm6 libxext6 libxrender1 && \
      rm -rf /var/lib/apt/lists/*; \
    else command -v python3 && command -v curl; fi

FROM runtime-base AS builder
WORKDIR /opt/vla-code
COPY project/ /opt/vla-code/
COPY offline-bundle.tar.gz /tmp/offline-bundle.tar.gz
RUN bash scripts/install_local.sh --bundle /tmp/offline-bundle.tar.gz \
      --bundle-dir /opt/vla-bundle --venv /opt/vla-env

FROM runtime-base AS final
ENV VLA_ENV=/opt/vla-env \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONNOUSERSITE=1
WORKDIR /opt/vla-code
COPY --from=builder /opt/vla-bundle/tools/python/ /opt/vla-bundle/tools/python/
COPY --from=builder /opt/vla-env/ /opt/vla-env/
COPY --from=builder /opt/vla-code/ /opt/vla-code/
CMD ["bash"]
