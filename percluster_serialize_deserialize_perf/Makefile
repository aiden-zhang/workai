#if compile kernel,add:-x cuda --cuda-gpu-arch=dlgput64  
all:
	dlcc  -std=c++14  main.cc -I$(DLICC_PATH)/../include/dlnne -I$(DLICC_PATH)/../include   -L$(DLICC_PATH)/../lib -lpthread -ldlnne 
clean:
	rm ./a.out ./*.o

help:
	echo $(DLICC_PATH)