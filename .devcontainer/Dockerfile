FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary dependencies.
RUN apt-get update && apt-get install -y \
    gfortran \
    build-essential \
    wget \
    curl \
    libblas3 \
    libblas-dev \
    liblapack3 \
    liblapack-dev \
    libmetis-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Julia.
ARG JULIA_VERSION=1.12.1
RUN JULIA_MAJOR=${JULIA_VERSION%.*} \
    && wget -q https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MAJOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia \
    && rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

# Add user.
ARG USER_NAME=developer
ARG USER_ID=1001
ARG USER_HOME="/home/$USER_NAME"

RUN adduser \
    --home "$USER_HOME" \
    --uid $USER_ID \
    --disabled-password \
    "$USER_NAME"

# Set the default user.
USER ${USER_NAME}

# Set the working directory to the user's home directory.
WORKDIR ${USER_HOME}

# Create a Julia package depot.
ENV JULIA_DEPOT_PATH="/home/developer/.julia"
ENV HSL_PATH="/home/developer/HSL"

# Optional: enable proprietary HSL linear solvers for Ipopt.
#
# This block expects the file
#   .devcontainer/HSL_jll.jl.v2024.11.28.tar.gz
# to be available in the build context (next to the Dockerfile).
#
# If this file is not available, comment out this entire block.
# The code will still run using open-source linear solvers, but may be slower and can yield slightly different results.
COPY --chown=${USER_NAME}:${USER_NAME} .devcontainer/HSL_jll.jl.v2024.11.28.tar.gz ${USER_HOME}/

RUN mkdir -p $HSL_PATH \
    && tar -xzf HSL_jll.jl.v2024.11.28.tar.gz \
        -C $HSL_PATH --strip-components=1 \
    && rm HSL_jll.jl.v2024.11.28.tar.gz

RUN julia -e 'using Pkg; Pkg.activate(); Pkg.develop(path=ENV["HSL_PATH"], shared=true); Pkg.precompile()'

# Copy Project.toml and Manifest.toml and install all packages.
COPY --chown=${USER_NAME}:${USER_NAME} Project.toml Manifest.toml ${USER_HOME}/
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()' \
    && rm Project.toml \
    && rm Manifest.toml