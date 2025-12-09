#!/bin/bash

# Update apt package list (needed in Docker where cache is cleaned)
apt-get update

# Install prerequisites
apt-get install -y lsb-release

# Navigate to parent directory
cd "$(dirname "$0")/.."

# Remove existing mininet directory if it exists
rm -rf mininet

# Clone Mininet repository
git clone https://github.com/mininet/mininet

# Navigate to the Mininet directory
cd mininet/util

# Run the Mininet installation script with the -a flag
./install.sh -a

# Start Open vSwitch service (required for Mininet)
echo "Starting Open vSwitch service..."
service openvswitch-switch start

echo "Mininet installation complete!"
echo "If OVS fails to start, run: service openvswitch-switch start"
