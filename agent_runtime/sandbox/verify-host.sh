#!/usr/bin/env bash
# Prove the sandbox host's isolation from INSIDE a real worker (decode fork F1). Read-only.
# Run as root on the host:  sudo bash verify-host.sh      Exit 0 only if every check passes.
set -uo pipefail

MAIN_VM_IP=10.0.0.49
PROXY_PORT=7601
HOST_IP=$(hostname -I | awk '{print $1}')
failures=0

check() {  # check <expect: open|blocked> <label> <host> <port>
  local expect=$1 label=$2 host=$3 port=$4 code result
  # Inside a hardened gVisor worker on the locked network, exactly as decode starts one.
  docker run --rm --runtime runsc --network bancs-only --cap-drop ALL --read-only --user 1000:1000 \
    decode-session:latest timeout 5 bash -c "</dev/tcp/$host/$port" 2>/dev/null
  code=$?
  # 0 = connected; 124 = timed out (a DROP); anything else = refused or rejected, which still proves
  # the PATH is open (nothing listens on the proxy port until F2).
  case $code in 0) result=open ;; 124) result=blocked ;; *) result="open (refused)" ;; esac
  [ "$result" = "open (refused)" ] && [ "$expect" = open ] && result=open
  if [ "$result" = "$expect" ]; then echo "PASS  $label: $result"; else echo "FAIL  $label: $result (expected $expect)"; failures=$((failures + 1)); fi
}

echo "== runtimes: $(docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}')"
echo "== gVisor in the worker: $(docker run --rm --runtime runsc decode-session:latest dmesg 2>/dev/null | head -1 || echo '?')"
check open    "BaNCS proxy ($MAIN_VM_IP:$PROXY_PORT)"   "$MAIN_VM_IP" "$PROXY_PORT"
check blocked "main VM SSH ($MAIN_VM_IP:22)"            "$MAIN_VM_IP" 22
check blocked "main VM HTTPS ($MAIN_VM_IP:443)"         "$MAIN_VM_IP" 443
check blocked "this host's SSH ($HOST_IP:22)"           "$HOST_IP" 22
check blocked "this host's runner ($HOST_IP:7700)"      "$HOST_IP" 7700
check blocked "the internet (1.1.1.1:443)"              1.1.1.1 443
check blocked "the BaNCS demo host directly (:443)"     "$(getent hosts demoapps.tcsbancs.com | awk '{print $1}')" 443
echo "== runner health: $(curl -s --max-time 5 "http://$HOST_IP:7700/health" || echo unreachable)"

[ "$failures" -eq 0 ] && echo "ALL CHECKS PASSED" || { echo "$failures CHECK(S) FAILED"; exit 1; }
