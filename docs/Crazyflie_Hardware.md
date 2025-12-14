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

## Run Model (with example radio addresses)

The number of URIs determines the number of drones which determines the policy (1 drone PPO or 2 drone MAPPO).

```bash
# For one drone:
python run_hardware.py --uri radio://0/80/2M/E7E7E7E701
    
# For two drones
python run_hardware.py --uri radio://0/80/2M/E7E7E7E701 radio://0/80/2M/E7E7E7E7E7
```

## Other possible Parameters (each has default, see `run_hardware.py`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--target` | Hover target [x, y, z] | `--target 0.0 0.0 0.5` |
| `--starting_pos_offset` | Offset drones by [x, y, z] since each drone sees [0, 0, 0] as starting position | `--starting_pos_offset 0.5 0.0 1.0` |
| `--dummy_policy` | Run a dummy policy (hover control) | `--dummy_policy` |
| `--duration` | Run time in seconds | `--duration 10.0` |
| `--bound_range` | Beyond this distance, the drone will receive a sub-hover safety control | `--bound_range 2.0` |
| `--land_at_time_left` | With < this time left, the drone will receive a sub-hover safety control | `--land_at_time_left 3.0` |
| `--control_rate` | Control frequency (Hz) | `--control_rate 50` |



# Physical Crazyflie Drone Documentation

## Relevant Logging Information (Sensor Data)
> Taken from [here](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/)
- In the `crazyflie-firmware` API, logs are accessible through the `crazyflie-lib-python` library as variables in the `LogConfig`

#### stateEstimate
- Estimation of position, velocity, and attitude (quaternion) in space

#### gyro 
- Angular velocity

## Connection
> Taken from the [Crazyflie API](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/), specifically the [logging groups and variables](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/)

- Two options, [asynchronous](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/) and [synchronous/blocking](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/syncCrazyflie/)
    - Will likely want to use the async constructor as the synchronous one is meant more for simple scripts

### Swarm
> Taken from the [Crazyflie swarm docs](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/swarm/)
- Could be used to manage our multiple agents, seems like you can spawn them as async or sync.
- Main functionality is running a single function across all of the drones in the swarm. Would need to pass through the loggers of all drones to each drone if we want them to see eachother in the global space.
