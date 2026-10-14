# Public image (no NGC login). For NGC: nvcr.io/nvidia/pytorch:24.07-py3
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Avoid some interactive prompts + make pip quieter/reproducible-ish
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Where your code will live inside the container
WORKDIR /workspace

# System deps (EGL/OpenGL + X11/xvfb for pyrender/pyglet in headless Docker)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      cmake build-essential \
      gosu \
      libegl1 libgles2 libgl1-mesa-glx libglvnd0 libglx0 \
      xvfb libx11-6 libxrender1 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

# Some base images ship a broken `/usr/local/bin/cmake` shim (from a partial pip install),
# which shadows `/usr/bin/cmake` and breaks builds that invoke `cmake` 
# Prefer the system cmake.
RUN rm -f /usr/local/bin/cmake || true

# Install Python deps first (better layer caching)
COPY assets /workspace/assets
COPY soma /workspace/soma
COPY tools /workspace/tools
COPY setup.cfg /workspace/setup.cfg
COPY setup.py /workspace/setup.py
COPY README.md /workspace/README.md
COPY pyproject.toml /workspace/pyproject.toml

# chumpy's build assumes 'pip' is in the build env; install it without isolation first
# chumpy uses inspect.getargspec (removed in Python 3.11); patch before any import
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-build-isolation chumpy \
 && find /opt/conda/lib/python3.11/site-packages/chumpy -name "*.py" -exec sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' {} \; \
 && python -m pip install .[smpl,anny]

RUN python -m pip install pyrender tqdm pyyaml imageio[ffmpeg]

# Use the docker-entrypoint script, to allow the docker to run as the actual user instead of root
COPY tools/docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint

# Default command (change to your entrypoint if you have one)
ENTRYPOINT ["docker-entrypoint"]
CMD ["bash"]

