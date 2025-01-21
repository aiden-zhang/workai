#!/bin/bash

MODULE="char_device"
DEVICE="/dev/char_device"

# 加载模块
echo "Loading module..."
sudo insmod ${MODULE}.ko
sleep 1

# 检查设备文件是否存在
if [ ! -e "$DEVICE" ]; then
    echo "Device file not found, creating manually..."
    major=$(dmesg | grep 'char_device: registered with major number' | awk '{print $NF}')
    sudo mknod $DEVICE c $major 0
    sudo chmod 666 $DEVICE
else
    echo "Device file found: $DEVICE"
fi

# 写入数据
echo "Testing write operation..."
echo "Hello, driver!" > $DEVICE

# 读取数据
echo "Testing read operation..."
echo "Reading from device:"
cat $DEVICE

# 验证内核日志
echo "Checking kernel logs..."
dmesg | tail

# 卸载模块
echo "Unloading module..."
sudo rmmod $MODULE

# 检查设备文件是否被删除
if [ -e "$DEVICE" ]; then
    echo "Cleaning up device file..."
    sudo rm $DEVICE
fi

echo "Test completed."
