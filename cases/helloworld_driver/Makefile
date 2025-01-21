obj-m += char_device.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	gcc  testioctl.c

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm a.out
