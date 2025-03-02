# Teleoperation Guide

The Doc focuses on the installation and usage of Leapmotion, Just follow the precedure to install and use leapmotion smoothly.
## Install LeapMotion
### Installation
#### DownLoad LeapMotion Installation File
First Download 'LeapMotion' Assets Filefolder from [Google Drive Link](https://drive.google.com/drive/folders/1EWH9zYQfBa96Z4JyimvUSBYOyW615JSg)

#### Installation
Installation precedure:
```bash
sudo apt-get install libgl1-mesa-glx
cd Your Leapmotion path/LeapDeveloperKit_2.3.1+31549_linux
sudo dpkg -i Leap-2.3.1+31549-x64.deb
```
Then create a new file for systemd
``` bash
sudo vi /lib/systemd/system/leapd.service
Add following content into file above

    >[Unit]
    >
    >Description=LeapMotion Daemon
    >
    >After=syslog.target
    >
    >[Service]
    >
    >Type=simple
    >
    >ExecStart=/usr/sbin/leapd
    >
    >[Install]
    >
    >WantedBy=multi-user.target
```

``` bash

sudo ln -s /lib/systemd/system/leapd.service /etc/systemd/system/leapd.service

sudo systemctl daemon-reload

sudo service leapd start

sudo apt-get install swig g++ libpython3-dev

cd Your Leamotion Path/leap-sdk-python3

sudo make install
```

###  LeapMotion Usage

1. connect leapmotion (<span style="color:red">Remember to keep this window open</span>)
```bash
sudo leapd
```
<span style="color:red">Remember to keep this window open</span>


### Verify installation
Open a new terminal and open Visualization
``` bash
Visualizer
```

Now if you can see a screen showing the hand skeleton and position, your installation is success.

# Start teleoperation
If you are in the simulator you can directly execute file
```bash
# Bimanual
python Teleoperation/Simulaton/DexBimanualCapture.py
# Right
python Teleoperation/Simulaton/DexLeftCapture.py
# Left
python Teleoperation/Simulaton/DexRightCapture.py
```
The trajectory file is stored in Assets/Replays

You can use replay script to replay it.
```bash
#Remember to change the path to trajectory file
# Bimanual
python Teleoperation/Simulaton/DexBimanualReplay.py
# Right
python Teleoperation/Simulaton/DexLeftReplay.py
# Left
python Teleoperation/Simulaton/DexRightReplay.py
```