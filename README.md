# xrctest

An example on how to establish an RDMA XRC (eXtended Reliable Connection).

## Compile 

* Refer to the [official document](https://docs.mellanox.com/display/MLNXOFEDv543030/Advanced+Transport).
  Check if your RNIC device and OFED driver support XRC.
  Try running `ibv_devinfo -v | grep XRC`.
  Normally, if you are using ConnectX-5/6, this should be OK.
* `cmake` >= 3.10 (you may use lower versions, modify `CMakeLists.txt`).
* `openmpi` installed, this is used in control plane.

## Run

* Establish a two-node MPI-available environment. Modify `hosts` and set the hostnames to proper ones.
* Run `run.sh` in your shell.
