# instant-NeuTex

## Usage

### Generate dataset by mitsuba3
1. get mitsuba3 and compile it
    ```bash
    git clone --recursive https://github.com/mitsuba-renderer/mitsuba3.git
    cd mitsuba3
    mkdir build
    cd build
    cmake -GNinja ..
    ninja
    ```
2. configuring mitsuba.conf  
    This file can be found in the build directory and will be created when executing CMake the first time.  
    Open mitsuba.conf and scroll down to the declaration of the enabled variants (around line 86):  
    ```bash
    "enabled": [
        "scalar_rgb", "cuda_rgb"
    ],
    ```
3. generate soft link of mitsuba3
   ```bash
   ln -s mitsuba3/build/mitsuba3 instant-NeuTex/
   ```
4. render dataset
   ```bash
   python instant-NeuTex/scene/generate_data.py --gpu -size 256
   ```

### Train instant-NeuTex
```bash
cd instant-NeuTex/run
bash bunny.sh 404
```

'404' is the name of this training process  
Install any missing library from pip

