#!/usr/bin/env bash
# Construye la imagen e imprime su digest, que es lo que se registra como
# procedencia. Sin digest la corrida no es reproducible.
set -euo pipefail

VERSION="${1:-0.1.0}"
TAG="opengames/trellis2:${VERSION}"

docker build -t "$TAG" "$(dirname "$0")"
echo
echo "imagen : $TAG"
echo "digest : $(docker image inspect "$TAG" --format '{{index .Id}}')"
echo "tamaño : $(docker image inspect "$TAG" --format '{{.Size}}' | awk '{printf "%.1f GB\n", $1/1e9}')"
