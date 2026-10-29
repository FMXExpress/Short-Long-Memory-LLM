#!/usr/bin/env bash
# Mem.Https against a local TLS server with throwaway certificates:
# a trusted chain must work; an untrusted CA, a wrong hostname and an
# expired certificate must all be refused.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/https_get
W=$(mktemp -d)
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$W"' EXIT
cd "$W"
q() { openssl "$@" >/dev/null 2>&1; }
q req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem -days 2 -subj "/CN=memagent test CA"
leaf() {  # name, SAN, days
  q req -newkey rsa:2048 -nodes -keyout "$1.key" -out "$1.csr" -subj "/CN=$2"
  printf 'subjectAltName=DNS:%s\n' "$2" > "$1.ext"
  q x509 -req -in "$1.csr" -CA ca.pem -CAkey ca.key -CAcreateserial -out "$1.pem" -days "$3" -extfile "$1.ext"
}
leaf good localhost 2
leaf wrong otherhost.example 2
# an already-expired leaf: valid from 3 days ago to 2 days ago
q req -newkey rsa:2048 -nodes -keyout old.key -out old.csr -subj "/CN=localhost"
printf 'subjectAltName=DNS:localhost\n' > old.ext
openssl x509 -req -in old.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out old.pem \
  -not_before "$(date -u -d '3 days ago' +%Y%m%d%H%M%SZ)" -not_after "$(date -u -d '2 days ago' +%Y%m%d%H%M%SZ)" \
  -extfile old.ext >/dev/null 2>&1 || q x509 -req -in old.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out old.pem -days -1 -extfile old.ext
echo "hello from the test server" > hello.txt

serve() { openssl s_server -quiet -WWW -accept "$1" -cert "$2.pem" -key "$2.key" >/dev/null 2>&1 & }
serve 18443 good; serve 18444 wrong; serve 18445 old
sleep 1
cd - >/dev/null
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"
fails=0
expect() {  # name, should-succeed (1/0), env CA file, url
  local out err rc=0
  err=$(SSL_CERT_FILE="$3" SSL_CERT_DIR=/nonexistent "$BIN" "$4" "$W/out.txt" 2>&1) || rc=$?
  if [ "$2" = 1 ] && [ $rc = 0 ] && grep -q "hello from the test server" "$W/out.txt"; then
    echo "  ok    $1"
  elif [ "$2" = 0 ] && [ $rc != 0 ]; then
    echo "  ok    $1 (refused: ${err:0:90})"
  else
    echo "  FAIL  $1 (rc=$rc ${err:0:120})"; fails=$((fails+1))
  fi
  rm -f "$W/out.txt"
}
expect "trusted CA + matching host"   1 "$W/ca.pem"                https://localhost:18443/hello.txt
expect "CA not trusted"               0 /root/.ccr/ca-bundle.crt   https://localhost:18443/hello.txt
expect "CA not trusted (empty store)" 0 /dev/null                  https://localhost:18443/hello.txt
expect "wrong hostname"               0 "$W/ca.pem"                https://localhost:18444/hello.txt
expect "expired certificate"          0 "$W/ca.pem"                https://localhost:18445/hello.txt
expect "plain http refused"           0 "$W/ca.pem"                http://localhost:18443/hello.txt
[ $fails = 0 ] && echo "all https checks passed" || { echo "$fails https check(s) failed"; exit 1; }
