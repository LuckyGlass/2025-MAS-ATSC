# 2025-MAS-ATSC
The individual course homework of Multi-Agent System, Peking University, 2025.

## Environment

## Install sumo

```shell
wget https://sumo.dlr.de/releases/1.11.0/sumo-src-1.11.0.tar.gz
tar -xf sumo*.gz
cd sumo*0
sudo apt-get update
sudo apt-get install cmake g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
export SUMO_HOME="$PWD"
mkdir build/cmake-build && cd build/cmake-build
sudo apt-get install libxerces-c-dev
```

You probably need to reinstall `libcurl4-openssl-dev`:
```shell
sudo apt-get remove libcurl4-openssl-dev
sudo apt-get install libcurl4-openssl-dev
```

Build:

```shell
cmake ../..
make -j$(nproc)
cd ../..
env | grep -i sumo
echo "export SUMO_HOME=$PWD" >> ~/.bashrc
echo "export PATH=$PWD/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

You can check the installation:

```shell
ls $SUMO_HOME/bin
sumo --version
```

## Install python environment

Critic packages: Python 3.11, Pytorch 2.7.0 (CUDA 11.8).

1. Create Conda environment with `python=3.11`.
2. Install `torch==2.7.0`.
3. The other packages are listed in `requirements.txt`: `pip install -r requirements.txt`.

## Reproduce the results

Please refer to `launch.py` and choose one of the `exp_name`. Execute the following command to reproduce the result.
```shell
python launch.py <exp_name>
```
