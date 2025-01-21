#include <linux/init.h>         // 模块初始化和清理宏
#include <linux/module.h>       // 核心模块头文件
#include <linux/kernel.h>       // 用于 printk
#include <linux/fs.h>           // 文件操作结构体
#include <linux/uaccess.h>      // 用于从用户空间复制数据
#include <linux/cdev.h>         // 字符设备结构体
#include <linux/device.h>       // class_create 和 device_create

#define DEVICE_NAME "char_device"
#define CLASS_NAME "char_class"
#define BUFFER_SIZE 256

// 定义 ioctl 命令
#define IOCTL_GET_BUFFER_SIZE _IOR('c', 1, int)
#define IOCTL_CLEAR_BUFFER    _IO('c', 2)
#define IOCTL_WRITE_STRING    _IOW('c', 3, char[BUFFER_SIZE])


static int major;                // 主设备号
static char device_buffer[BUFFER_SIZE]; // 设备缓冲区
static struct cdev char_dev;     // 字符设备结构体
static struct class *char_class = NULL; // 设备类
static struct device *char_device = NULL; // 设备

// 打开设备
static int char_device_open(struct inode *inode, struct file *file) {
    printk(KERN_INFO "char_device: device opened\n");
    return 0;
}

// 关闭设备
static int char_device_release(struct inode *inode, struct file *file) {
    printk(KERN_INFO "char_device: device closed\n");
    return 0;
}

// 从设备读取数据
static ssize_t char_device_read(struct file *file, char __user *buf, size_t count, loff_t *ppos) {
    size_t available_size = BUFFER_SIZE - *ppos; // 可读取的剩余数据量
    size_t read_size = count > available_size ? available_size : count;

    if (read_size == 0) {
        printk(KERN_INFO "char_device: no more data to read\n");
        return 0; // 已到达文件末尾
    }

    if (copy_to_user(buf, device_buffer + *ppos, read_size)) {
        return -EFAULT; // 用户空间复制失败
    }

    *ppos += read_size; // 更新偏移量
    printk(KERN_INFO "char_device: read %zu bytes\n", read_size);
    return read_size;
}

// 向设备写入数据
static ssize_t char_device_write(struct file *file, const char __user *buf, size_t count, loff_t *ppos) {
    size_t available_size = BUFFER_SIZE - *ppos; // 可写入的剩余空间
    size_t write_size = count > available_size ? available_size : count;

    if (write_size == 0) {
        printk(KERN_INFO "char_device: no space left to write\n");
        return -ENOMEM; // 空间不足
    }

    if (copy_from_user(device_buffer + *ppos, buf, write_size)) {
        return -EFAULT; // 用户空间复制失败
    }

    *ppos += write_size; // 更新偏移量
    printk(KERN_INFO "char_device: wrote %zu bytes\n", write_size);
    return write_size;
}

// ioctl 实现
static long char_device_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    int buffer_size;
    char user_buffer[BUFFER_SIZE];

    switch (cmd) {
        case IOCTL_GET_BUFFER_SIZE:
            buffer_size = strlen(device_buffer); // 获取缓冲区大小
            if (copy_to_user((int __user *)arg, &buffer_size, sizeof(buffer_size))) {
                return -EFAULT;
            }
            printk(KERN_INFO "char_device: ioctl GET_BUFFER_SIZE -> %d\n", buffer_size);
            break;

        case IOCTL_CLEAR_BUFFER:
            memset(device_buffer, 0, BUFFER_SIZE); // 清空缓冲区
            printk(KERN_INFO "char_device: ioctl CLEAR_BUFFER -> buffer cleared\n");
            break;

        case IOCTL_WRITE_STRING:
            if (copy_from_user(user_buffer, (char __user *)arg, BUFFER_SIZE)) {
                return -EFAULT;
            }
            strncpy(device_buffer, user_buffer, BUFFER_SIZE - 1);
            device_buffer[BUFFER_SIZE - 1] = '\0'; // 确保字符串以 NULL 结尾
            printk(KERN_INFO "char_device: ioctl WRITE_STRING -> %s\n", device_buffer);
            break;

        default:
            printk(KERN_WARNING "char_device: invalid ioctl command\n");
            return -EINVAL;
    }

    return 0;
}

// 文件操作接口
static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = char_device_open,
    .release = char_device_release,
    .read = char_device_read,
    .write = char_device_write,
    .unlocked_ioctl = char_device_ioctl, // 添加 ioctl 函数
};

// 初始化模块
static int __init char_device_init(void) {
    dev_t dev;

    // 分配主设备号
    if (alloc_chrdev_region(&dev, 0, 1, DEVICE_NAME) < 0) {
        printk(KERN_ALERT "char_device: failed to allocate major number\n");
        return -1;
    }
    major = MAJOR(dev);
    printk(KERN_INFO "char_device: registered with major number %d\n", major);

    // 初始化字符设备
    cdev_init(&char_dev, &fops);
    char_dev.owner = THIS_MODULE;

    // 将字符设备添加到内核
    if (cdev_add(&char_dev, dev, 1) < 0) {
        unregister_chrdev_region(dev, 1);
        printk(KERN_ALERT "char_device: failed to add cdev\n");
        return -1;
    }

    // 创建设备类
    char_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(char_class)) {
        cdev_del(&char_dev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_ALERT "char_device: failed to create class\n");
        return PTR_ERR(char_class);
    }

    // 创建设备文件
    char_device = device_create(char_class, NULL, dev, NULL, DEVICE_NAME);
    if (IS_ERR(char_device)) {
        class_destroy(char_class);
        cdev_del(&char_dev);
        unregister_chrdev_region(dev, 1);
        printk(KERN_ALERT "char_device: failed to create device\n");
        return PTR_ERR(char_device);
    }

    printk(KERN_INFO "char_device: device initialized successfully\n");
    return 0;
}

// 清理模块
static void __exit char_device_exit(void) {
    dev_t dev = MKDEV(major, 0);

    // 删除设备文件
    device_destroy(char_class, dev);

    // 销毁设备类
    class_destroy(char_class);

    // 移除字符设备
    cdev_del(&char_dev);

    // 释放主设备号
    unregister_chrdev_region(dev, 1);

    printk(KERN_INFO "char_device: device exited\n");
}

module_init(char_device_init);
module_exit(char_device_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Aiden");
MODULE_DESCRIPTION("A simple character device driver with automatic device file creation.");
MODULE_VERSION("1.1");
