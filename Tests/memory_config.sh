#!/bin/bash
# Jetson Nano Memory Optimization for TensorFlow
# Run with sudo: sudo bash jetson-memory-config.sh

echo "Jetson Nano Memory Configuration for TensorFlow 1.15"
echo "===================================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo)"
  exit 1
fi

# Function to get current memory allocation
check_memory() {
  echo "Current memory status:"
  free -h
  echo ""
  echo "Current swap configuration:"
  swapon --show
  echo ""
}

# Function to set up larger swap file
setup_swap() {
  echo "Setting up 4GB swap file..."
  
  # Check if swap exists
  SWAP_EXISTS=$(swapon --show | grep -c "/swap")
  
  # Remove existing swap if found
  if [ $SWAP_EXISTS -gt 0 ]; then
    echo "Removing existing swap file..."
    swapoff -a
    rm -f /swap
  fi
  
  # Create new swap file (4GB)
  echo "Creating 4GB swap file (this may take a minute)..."
  fallocate -l 4G /swap
  chmod 600 /swap
  mkswap /swap
  swapon /swap
  
  # Add to fstab if not already there
  if ! grep -q "^/swap" /etc/fstab; then
    echo "Adding swap to fstab for persistence across reboots..."
    echo "/swap swap swap defaults 0 0" >> /etc/fstab
  fi
  
  # Verify swap is active
  echo "Swap configuration complete:"
  swapon --show
}

# Function to adjust swappiness 
adjust_swappiness() {
  echo "Adjusting swappiness for TensorFlow performance..."
  
  # Set swappiness to 10 (prefer RAM, but use swap when needed)
  echo 10 > /proc/sys/vm/swappiness
  
  # Make persistent across reboots
  if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
    echo "vm.swappiness=10" >> /etc/sysctl.conf
  else
    sed -i 's/vm.swappiness=.*/vm.swappiness=10/' /etc/sysctl.conf
  fi
  
  # Apply changes
  sysctl -p
  
  echo "Current swappiness: $(cat /proc/sys/vm/swappiness)"
}

# Function to configure TensorFlow memory growth
configure_tensorflow_growth() {
  echo "Configuring environment for TensorFlow memory growth..."
  
  # Create the environment file
  TF_ENV_FILE="/etc/profile.d/tensorflow-memory.sh"
  
  echo "# TensorFlow memory optimization" > $TF_ENV_FILE
  echo "export TF_FORCE_GPU_ALLOW_GROWTH=true" >> $TF_ENV_FILE
  echo "export TF_GPU_ALLOCATOR=cuda_malloc_async" >> $TF_ENV_FILE
  echo "export TF_CPP_MIN_LOG_LEVEL=0" >> $TF_ENV_FILE
  
  # Make it executable
  chmod +x $TF_ENV_FILE
  
  echo "Environment configured. Changes will take effect after reboot."
}

# Function to optimize Jetson power mode
optimize_power_mode() {
  echo "Setting Jetson Nano to 10W power mode for better performance..."
  
  # Check if nvpmodel command exists
  if command -v nvpmodel &> /dev/null; then
    # Set to 10W mode (mode 0)
    nvpmodel -m 0
    
    # Maximize clock frequencies
    if command -v jetson_clocks &> /dev/null; then
      jetson_clocks
    fi
    
    echo "Power mode set to 10W with maximum clock speeds."
  else
    echo "nvpmodel not found. Skipping power optimization."
  fi
}

# Main execution
check_memory
setup_swap
adjust_swappiness
configure_tensorflow_growth
optimize_power_mode

echo ""
echo "Memory optimization complete!"
echo "For best results, please reboot your Jetson Nano:"
echo "sudo reboot"
