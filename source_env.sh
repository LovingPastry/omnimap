conda deactivate
conda deactivate
conda activate InfoFlow
source /opt/ros/noetic/setup.bash
# export AM_I_DOCKER=False
# export BUILD_WITH_CUDA=True
# export CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:/home/fuyx/anaconda3/envs/omnimap/bin:$PATH
# export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH"
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export ROS_MASTER_URI=http://172.19.93.154:11311
export ROS_HOSTNAME=172.19.73.218
export ROS_IP=172.19.73.218

