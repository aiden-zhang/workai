#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string.h>

#define DEVICE "/dev/char_device"

// 定义与驱动一致的 ioctl 命令
#define IOCTL_GET_BUFFER_SIZE _IOR('c', 1, int)
#define IOCTL_CLEAR_BUFFER    _IO('c', 2)
#define IOCTL_WRITE_STRING    _IOW('c', 3, char[256])

int main() {
    int fd = open(DEVICE, O_RDWR);
    if (fd < 0) {
        perror("Failed to open the device");
        return 1;
    }

    // 获取缓冲区大小
    int buffer_size;
    if (ioctl(fd, IOCTL_GET_BUFFER_SIZE, &buffer_size) == 0) {
        printf("Buffer size: %d\n", buffer_size);
    } else {
        perror("Failed to get buffer size");
    }

    // 写入字符串
    char message[] = "Hello, ioctl!";
    if (ioctl(fd, IOCTL_WRITE_STRING, message) == 0) {
        printf("Message written to buffer: %s\n", message);
    } else {
        perror("Failed to write string");
    }

    // 清空缓冲区
    if (ioctl(fd, IOCTL_CLEAR_BUFFER) == 0) {
        printf("Buffer cleared.\n");
    } else {
        perror("Failed to clear buffer");
    }

    close(fd);
    return 0;
}
