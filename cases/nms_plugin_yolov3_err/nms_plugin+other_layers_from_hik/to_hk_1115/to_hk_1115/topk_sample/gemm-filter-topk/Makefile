


CXX = dlcc
AR = ar cqs
SDK_PATH = $(DLICC_PATH)/../

TARGET ?= libgft.so
TARGETA ?= libgft.a

MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR = $(dir $(MAKEFILE_PATH))
CXX_FLAGS=-std=c++14 -I$(MAKEFILE_DIR) -I$(SDK_PATH)/include/dlnne -fPIC
DLFLAGS = --cuda-gpu-arch=dlgpuc64 -D__DLGPUC64__

VPATH = $(MAKEFILE_DIR)
CC_FILE += $(wildcard $(MAKEFILE_DIR)/utils/*.cpp)
CC_OBJS += $(subst .cpp,.occ, $(CC_FILE))

CU_FILE += $(wildcard $(MAKEFILE_DIR)/topk/*.cu)
CU_OBJS += $(subst .cu,.ocu, $(CU_FILE))

RUN_CUS += $(wildcard $(MAKEFILE_DIR)/*.cu)
RUN_OBJS += $(subst .cu,.ocu, $(RUN_CUS))
RUN_EXES += $(notdir $(patsubst %.ocu,%.run, $(RUN_OBJS)))

LINKS += -lpthread -lcurt -lcudnn

%.occ: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

%.ocu: %.cu
	$(CXX) $(CXX_FLAGS) $(DLFLAGS) -DSRC_DIR=\"-I$(MAKEFILE_DIR)\" -o $@ -c $<

%.run: %.ocu $(CC_OBJS) ${CU_OBJS}
	$(CXX) -o $@ $< $(CC_OBJS) ${CU_OBJS} ${LINKS}

all: $(RUN_EXES)

shared: $(CC_OBJS) ${CU_OBJS}
	$(CXX) -shared -o $(TARGET) $(CC_OBJS) ${CU_OBJS}

static: $(CC_OBJS) ${CU_OBJS}
	$(AR) -o $(TARGETA) $(CC_OBJS) ${CU_OBJS}

fatbin: all
	./genfatbin.run
	
clean:
	rm -f $(CC_OBJS)
	rm -f $(CU_OBJS)
	rm -f $(RUN_OBJS)

distclean: clean
	rm -f $(RUN_EXES)
	rm -f $(TARGET)
	rm -f $(TARGETA)
	rm -f kernels.bc
	rm -f kernels.fatbin
