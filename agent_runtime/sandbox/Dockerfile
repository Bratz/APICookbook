# The Build & run session image: what a visitor's code runs in (decode fork F1).
# Built on the sandbox host (aarch64) by .github/workflows/sandbox-host-setup.yml as decode-session:latest.
#
# The worker runs hardened (SANDBOX_DOCKER_HARDENED): uid 1000, read-only root, no capabilities and
# no network except the BaNCS proxy. So everything a generated client needs is baked in here: the
# container cannot apt-get or pip install at run time.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      bash curl ca-certificates git jq nodejs openjdk-17-jdk-headless \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir requests==2.32.3 httpx==0.28.1 \
 && useradd --uid 1000 --create-home --shell /bin/bash sandbox \
 && git config --system user.name "decode" \
 && git config --system user.email "decode@localhost" \
 && git config --system --add safe.directory /workspace

USER 1000:1000
WORKDIR /workspace
