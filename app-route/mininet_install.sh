#!/bin/bash
cd ..
# Clone Mininet repository
git clone https://github.com/mininet/mininet

# Navigate to the Mininet directory
cd mininet

# Navigate to the util directory
cd util

# Run the Mininet installation script with the -a flag
sudo ./install.sh -a