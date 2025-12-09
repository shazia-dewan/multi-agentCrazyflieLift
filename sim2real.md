# sim2real for Crazyflie 2.0 MARL

## Connection to the Crazyflie drones
### Requirements:
- `cfclient`
- crazyflie 2.1 drone
- Flowdeck 2.0 
- microUSB -> USB

### Computer Setup
You will likely need to install `cfclient` before running the code. This can be installed with:
```
pip install cfclient
```
If you are using Linux, you will also need to set up the Crazyflie Radio device drivers. The instructions can be found [here](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/).

### Initial configuration
On a brand new drone, the default radio address is 0xE7E7E7E7E7. If you are using existing lab hardware, expect that to be different to avoid colliding addresses. To determine the address of a specific drone, you can open `cfclient`, connect the drone to your computer via the microUSB to USB cable. Using the menubar, go to Connect -> Bootloader. With the interface set to `usb://0` (might have to hit 'Scan' for this to appear) and the address empty, hit Connect. Back in the main window, you can go to Connect -> Configure 2.x, and there you can view and update the address of your drone. It is also important to take note of the `Radio channel` and `Radio bandwidth`, as you will need those to assemble your radio URI.

## Preparing the Physical Environment

## Running a Model
Now that you have your Crazyflie radio URL(s), you can run the the hardware runner script.

If you are only using a single drone, you can run the following:
```
python run_hardware_deployment.py --model_path </path/to/model> --uri radio://0/<radio_channel>/<radio_bandwidth>/<radio_address>
```

If you are planning on running a multi-agent model, you can run the following:
```
python run_hardware_deployment.py --model_path </path/to/model> --num_drones <num_drones> --uri radio://0/<radio_channel>/<radio_bandwidth>/<radio_address> radio://0/<radio_channel>/<radio_bandwidth>/<radio_address>
```

Fine-tuning between different drones is also required. In our experimentation we noticed a different response from the same thrust value across multiple drones. 