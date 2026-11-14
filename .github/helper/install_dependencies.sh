#!/bin/bash
set -e

echo "Setting Up System Dependencies..."

sudo apt-get update
sudo apt-get remove -y mysql-server mysql-client
sudo apt-get install -y --no-install-recommends \
  build-essential libmariadb-dev mariadb-client pkg-config
