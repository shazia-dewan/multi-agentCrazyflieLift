# Physical Crazyflie Drone Documentation

## Logging Information (Sensor Data)
> Taken from [here](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/)
- In the `crazyflie-firmware` API, logs are accessible through the `crazyflie-lib-python` library as variables in the `LogConfig`

### Logs of interest to us

#### acceleration
- Acceleration of the CF on a specific axis

| Name  | Core | Type | Description |
|-------|------|-----------|------------------------------|
| acc.x | Core | LOG_FLOAT | Acceleration in X [Gs].|
| acc.y | Core | LOG_FLOAT | Acceleration in Y [Gs].|
| acc.z | Core | LOG_FLOAT | Acceleration in Z [Gs].|

#### stateEstimate
- Estimation of position, velocity, acceleration, and attitude in space

| Name| Core | Type| Description|
|-------------------|------|-----------|-----------------------------------------------------------------------------|
| stateEstimate.x| Core | LOG_FLOAT | The estimated position of the platform in the global reference frame, X [m].|
| stateEstimate.y| Core | LOG_FLOAT | The estimated position of the platform in the global reference frame, Y [m].|
| stateEstimate.z| Core | LOG_FLOAT | The estimated position of the platform in the global reference frame, Z [m].|
| stateEstimate.vx| Core | LOG_FLOAT | The velocity of the Crazyflie in the global reference frame, X [m/s].|
| stateEstimate.vy| Core | LOG_FLOAT | The velocity of the Crazyflie in the global reference frame, Y [m/s].|
| stateEstimate.vz| Core | LOG_FLOAT | The velocity of the Crazyflie in the global reference frame, Z [m/s].|
| stateEstimate.ax| Core | LOG_FLOAT | The acceleration of the Crazyflie in the global reference frame, X [Gs].|
| stateEstimate.ay| Core | LOG_FLOAT | The acceleration of the Crazyflie in the global reference frame, Y [Gs].|
| stateEstimate.az| Core | LOG_FLOAT | The acceleration of the Crazyflie in the global reference frame, without considering gravity, Z [Gs].|
| stateEstimate.roll| Core | LOG_FLOAT | Attitude, roll angle [deg].|
| stateEstimate.pitch| Core | LOG_FLOAT | Attitude, pitch angle (legacy CF2 body coordinate system, where pitch is inverted) [deg].|
| stateEstimate.yaw| Core | LOG_FLOAT | Attitude, yaw angle [deg].|

#### gyro 
- Rotation around a specific axis, might be redundant with estimated attitude in `stateEstimate`

| Name| Core | Type| Description|
|--------|------|-----------|-------------------------------------------------------------------|
| gyro.x | Core | LOG_FLOAT | Angular velocity (rotation) around the X-axis, after filtering [deg/s].|
| gyro.y | Core | LOG_FLOAT | Angular velocity (rotation) around the Y-axis, after filtering [deg/s].|
| gyro.z | Core | LOG_FLOAT | Angular velocity (rotation) around the Z-axis, after filtering [deg/s].|

## Connection
> Taken from [here](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/)

- Two options, [asynchronous](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/) and [synchronous/blocking](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/syncCrazyflie/)
    - will likely want to use the async constructor as the synchronous one is meant more for simple scripts
- simply connect to the URI (dependent on method of connection, example code highlights radio)

### Swarm
> Taken from [here](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/api/cflib/crazyflie/swarm/)
- Could be used to manage our multiple agents, seems like you can spawn them as async or sync.
- Main functionality is running a single function across all of the drones in the swarm. Would need to pass through the loggers of all drones to each drone if we want them to see eachother in the global space.

