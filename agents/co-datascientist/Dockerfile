FROM mlebench-env

# Standard directory setup for MLE-bench agent
ARG DATA_DIR=/home/data
ARG BASELINE_DIR=/home/agent/baselines
ARG SUBMISSION_DIR=/home/submission
ARG LOGS_DIR=/home/logs
ARG CODE_DIR=/home/code
ARG AGENT_DIR=/home/agent

# Set environment variables
ENV DATA_DIR=${DATA_DIR}
ENV BASELINE_DIR=${BASELINE_DIR}
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ENV LOGS_DIR=${LOGS_DIR}
ENV CODE_DIR=${CODE_DIR}
ENV AGENT_DIR=${AGENT_DIR}
ENV PYTHONPATH="${AGENT_DIR}:${AGENT_DIR}/adapter"
# Set mlebench cache to writable location
ENV MLEBENCH_CACHE_DIR="${SUBMISSION_DIR}/.mlebench_cache"

# Create required directories
RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR} ${BASELINE_DIR}

ARG CONDA_ENV_NAME=agent

# Create conda environment with Python 3.12
RUN conda create -n ${CONDA_ENV_NAME} python=3.12 -y

# Copy the engine code for installation (when building from parent context)
# Expects build context to be Co-DataScientist_ root
COPY co-datascientist-engine /tmp/co-datascientist-engine

# Install co-datascientist-engine as a package from local copy
RUN conda run -n ${CONDA_ENV_NAME} pip install /tmp/co-datascientist-engine && \
    rm -rf /tmp/co-datascientist-engine && \
    conda clean -afy

# Copy mlebench for grading functionality
COPY mle-bench /tmp/mle-bench

# Install mlebench and additional dependencies needed by baselines and iteration tracking
RUN conda run -n ${CONDA_ENV_NAME} pip install \
    /tmp/mle-bench \
    torch torchvision pandas scikit-learn smolagents openai litellm python-dotenv matplotlib && \
    rm -rf /tmp/mle-bench && \
    conda clean -afy

# Copy adapter orchestration code (NOT the engine - it's installed as a package!)
COPY mle-bench/agents/co-datascientist/adapter/ ${AGENT_DIR}/adapter/

# Copy local_runner utilities (not part of the main package)
COPY co-datascientist-engine/src/local_runner/ ${AGENT_DIR}/local_runner/

# Copy baselines for competitions
COPY mle-bench/agents/co-datascientist/baselines/ ${BASELINE_DIR}/

# Copy configuration files
COPY mle-bench/agents/co-datascientist/start.sh ${AGENT_DIR}/
COPY mle-bench/agents/co-datascientist/config.yaml ${AGENT_DIR}/
COPY mle-bench/agents/co-datascientist/container.env ${AGENT_DIR}/.env

# Make start script executable
RUN chmod +x ${AGENT_DIR}/start.sh

# The container will be run with the start.sh script as entry point

