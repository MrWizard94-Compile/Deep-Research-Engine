FROM ubuntu:24.04

# Prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Update system base packages, tools, Java JDK, and Python runtimes
RUN apt-get update && apt-get install -y curl build-essential openjdk-21-jdk python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Install highly requested Python computational, caching, and data processing libraries
RUN apt-get update && apt-get install -y python3-redis python3-numpy python3-pandas python3-scipy python3-requests python3-zmq && rm -rf /var/lib/apt/lists/*

# Install official production-grade Rust via the direct shell script endpoint
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
