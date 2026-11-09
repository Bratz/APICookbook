#!/usr/bin/env bash
# Set up the Build & run sandbox host (decode fork F1). Idempotent: safe to re-run.
# Run as root on the host:  sudo REPO_SHA=<commit> bash setup-host.sh
# (.github/workflows/sandbox-host-setup.yml does this over SSH.) The runner's secrets are written
# separately, from the workflow, into /etc/decode/runner.env.
set -euo pipefail

MAIN_VM_IP=10.0.0.49         # cookbook-api: the runner's only client, and the BaNCS proxy's host
PROXY_PORT=7601              # the BaNCS proxy on the main VM (F2): the containers' ONLY destination
RUNNER_PORT=7700             # the runner, on this host's private address
SANDBOX_NET=bancs-only       # the docker network the workers run on
SANDBOX_BRIDGE=br-bancs
SANDBOX_SUBNET=172.30.0.0/24
RUN_USER=ubuntu              # uid 1000: owns the bind-mounted Workspaces, as the workers' uid 1000 does
REPO_URL=https://github.com/Bratz/APICookbook.git
REPO_DIR=/opt/apicookbook
REPO_SHA=${REPO_SHA:?set REPO_SHA to the commit to deploy}
export DEBIAN_FRONTEND=noninteractive

step() { echo; echo "== $*"; }

step "packages: docker, git"
apt-get update -q
apt-get install -yq docker.io git curl ca-certificates gnupg jq

step "gVisor (runsc) as a docker runtime"
if ! command -v runsc >/dev/null; then
  curl -fsSL https://gvisor.dev/archive.key | gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" \
    > /etc/apt/sources.list.d/gvisor.list
  apt-get update -q && apt-get install -yq runsc
fi
runsc install   # adds the "runsc" runtime to /etc/docker/daemon.json (idempotent)
systemctl restart docker
usermod -aG docker "$RUN_USER"

step "the locked network ($SANDBOX_NET -> only $MAIN_VM_IP:$PROXY_PORT)"
docker network inspect "$SANDBOX_NET" >/dev/null 2>&1 || docker network create --driver bridge \
  --subnet "$SANDBOX_SUBNET" -o com.docker.network.bridge.name="$SANDBOX_BRIDGE" "$SANDBOX_NET"

cat > /usr/local/sbin/decode-firewall.sh <<EOF
#!/usr/bin/env bash
# Re-applied after every docker start by decode-firewall.service.
set -euo pipefail
# Workers on $SANDBOX_BRIDGE may reach ONLY the BaNCS proxy: no internet, no other address.
iptables -N DECODE-SANDBOX 2>/dev/null || iptables -F DECODE-SANDBOX
iptables -A DECODE-SANDBOX -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
iptables -A DECODE-SANDBOX -d $MAIN_VM_IP -p tcp --dport $PROXY_PORT -j RETURN
iptables -A DECODE-SANDBOX -j DROP
iptables -C DOCKER-USER -i $SANDBOX_BRIDGE -j DECODE-SANDBOX 2>/dev/null \
  || iptables -I DOCKER-USER 1 -i $SANDBOX_BRIDGE -j DECODE-SANDBOX
# ...and nothing on this host itself (its sshd, the runner).
iptables -C INPUT -i $SANDBOX_BRIDGE -j DROP 2>/dev/null || iptables -I INPUT 1 -i $SANDBOX_BRIDGE -j DROP
# The runner answers only cookbook-api (Oracle's INPUT chain rejects everything else anyway).
iptables -C INPUT -p tcp -s $MAIN_VM_IP --dport $RUNNER_PORT -j ACCEPT 2>/dev/null \
  || iptables -I INPUT 2 -p tcp -s $MAIN_VM_IP --dport $RUNNER_PORT -j ACCEPT
EOF
chmod 755 /usr/local/sbin/decode-firewall.sh
cat > /etc/systemd/system/decode-firewall.service <<'EOF'
[Unit]
Description=Build & run sandbox firewall (decode fork)
After=docker.service
Requires=docker.service
PartOf=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/decode-firewall.sh

[Install]
WantedBy=multi-user.target docker.service
EOF
systemctl daemon-reload
systemctl enable decode-firewall.service
systemctl restart decode-firewall.service

step "code at $REPO_SHA, and its Python env"
if [ ! -d "$REPO_DIR/.git" ]; then git clone -q "$REPO_URL" "$REPO_DIR"; fi
git -C "$REPO_DIR" fetch -q origin
git -C "$REPO_DIR" reset -q --hard "$REPO_SHA"
chown -R "$RUN_USER:$RUN_USER" "$REPO_DIR"
sudo -u "$RUN_USER" -H bash -lc '
  set -e
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  cd '"$REPO_DIR"'/agent_runtime && uv sync --frozen --no-dev -q
'

step "session image"
docker build -q -t decode-session:latest "$REPO_DIR/agent_runtime/sandbox"

step "runner service"
install -d -o "$RUN_USER" -g "$RUN_USER" -m 750 /var/lib/decode /var/lib/decode/runs
PRIVATE_IP=$(hostname -I | awk '{print $1}')
cat > /etc/systemd/system/decode-runner.service <<EOF
[Unit]
Description=Build & run runner (decode fork)
After=docker.service decode-firewall.service
Requires=docker.service decode-firewall.service

[Service]
User=$RUN_USER
EnvironmentFile=/etc/decode/runner.env
WorkingDirectory=$REPO_DIR/agent_runtime
ExecStart=$REPO_DIR/agent_runtime/.venv/bin/python -m decode.runner --host $PRIVATE_IP --port $RUNNER_PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable decode-runner.service
if [ -f /etc/decode/runner.env ]; then systemctl restart decode-runner.service; fi
echo "setup done (runner on $PRIVATE_IP:$RUNNER_PORT)"
