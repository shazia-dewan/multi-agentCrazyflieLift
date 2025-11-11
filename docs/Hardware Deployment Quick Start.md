# Hardware Deployment Quick Start

Quick reference for running trained models on real Crazyflie drones.

## Installation

```bash
# Install Crazyflie library
pip install cflib

# Linux only: Install udev rules
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-crazyradio.rules # https://wiki.geeetech.com/index.php/Installation
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Run Model

```bash
# Single drone
python run_hardware_deployment.py \
    --model_path ../rl_models/ppo_model.pt \
    --uri radio://0/80/2M/E7E7E7E7E7

# Multiple drones
python run_hardware_deployment.py \
    --model_path ../rl_models/mappo_model.pt \
    --uri radio://0/80/2M/E7E7E7E7E7 radio://0/80/2M/E7E7E7E7E8 \
    --num_drones 2
```

## Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model_path` | Path to .pt model | `../rl_models/ppo_model.pt` |
| `--uri` | Drone URI(s) | `radio://0/80/2M/E7E7E7E7E7` |
| `--target` | Target position [x y z] | `--target 1.0 0.5 1.5` |
| `--duration` | Run time in seconds | `--duration 60` |
| `--control_rate` | Control frequency (Hz) | `--control_rate 100` |