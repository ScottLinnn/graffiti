# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

# FROM UBUNTU
# RUN apt-get update && \
#     DEBIAN_FRONTEND=noninteractive apt-get -y install gcc mono-mcs && \
#     rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=3.9.1
FROM python:${PYTHON_VERSION}-slim as base

FROM python:${PYTHON_VERSION}-slim as git
RUN \
  # --mount=type=cache,target=/var/cache/apt \
  apt-get update && apt-get install -y git

FROM git as clone
RUN git clone --depth=1 https://github.com/opencv/opencv.git
RUN git clone --depth=1 https://github.com/opencv/opencv_contrib

FROM base as build
COPY --from=clone /opencv /opencv
COPY --from=clone /opencv_contrib /opencv_contrib
RUN \
  # --mount=type=cache,target=/var/cache/apt \
  apt-get update && apt-get install -y cmake ccache g++
RUN python3 -m pip install numpy==1.26.0

# use ccache (make it appear in path earlier then /usr/bin/gcc etc)
# https://stackoverflow.com/questions/39650056/using-ccache-when-building-inside-of-docker
RUN for p in gcc g++ cc c++; do ln -vs /usr/bin/ccache /usr/local/bin/$p;  done

RUN mkdir build

WORKDIR /app/build

ENV CCACHE_DIR=/mnt/ccache

RUN cmake -DOPENCV_ENABLE_NONFREE=ON \
          -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
          -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF \
          /opencv

RUN --mount=type=cache,target=$CCACHE_DIR \
    ccache -s && make -j8 install

FROM base as final

WORKDIR /app

RUN \
  --mount=type=cache,target=/var/cache/apt \
  apt-get update && apt-get install -y curl

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

COPY --from=build /usr/local/lib/python3.9/site-packages/cv2/ \
                  /usr/local/lib/python3.9/site-packages/cv2/
COPY --from=build /usr/local/lib \
                  /usr/local/lib

RUN chown appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

WORKDIR /app

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8000
EXPOSE 9099

# Run the application.
CMD python benchmark.py --feature_extraction_algo=orb
# CMD python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
