


CC = gcc
CXX = g++
DLCC = dlcc

TARGET ?= AETest

MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR = $(dir $(MAKEFILE_PATH))

CU_FILE += $(wildcard $(MAKEFILE_DIR)/*.cu)
CU_OBJS += $(subst .cu,.ocu, $(CU_FILE))

CPP_FILE += $(wildcard $(MAKEFILE_DIR)/*.cpp)
CPP_FILE += $(wildcard $(MAKEFILE_DIR)/utils/*.cpp)
CPP_OBJS +=$(subst .cpp,.opp, $(CPP_FILE))

SDK_DIR=$(subst /bin,,$(DLICC_PATH))
CXX_FLAGS += -std=c++14 -I$(MAKEFILE_DIR) -I$(MAKEFILE_DIR)/../gemm-filter-topk -I$(MAKEFILE_DIR)/googletest/googletest/include -I$(SDK_DIR)/include -I$(SDK_DIR)/include/dlnne
DLCC_FLAGS += --cuda-gpu-arch=dlgpuc64
LINKS += -L$(SDK_DIR)/lib -Wl,-rpath,$(SDK_DIR)/lib -lpthread -lcurt -lcudnn -lgtest -L. -lgft

%.opp: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

%.ocu: %.cu
	$(DLCC) $(CXX_FLAGS) $(DLCC_FLAGS) -o $@ -c $<

all: $(CU_OBJS) $(CPP_OBJS)
	make shared -f $(MAKEFILE_DIR)/../gemm-filter-topk/Makefile
	make fatbin -f $(MAKEFILE_DIR)/../gemm-filter-topk/Makefile
	$(CXX) -o $(TARGET) $(CU_OBJS) $(CPP_OBJS) ${LINKS}

clean:
	make clean -f $(MAKEFILE_DIR)/../gemm-filter-topk/Makefile
	rm -f $(CU_OBJS)
	rm -f $(CPP_OBJS)

distclean: clean
	make distclean -f $(MAKEFILE_DIR)/../gemm-filter-topk/Makefile
	rm -rf $(TARGET)
