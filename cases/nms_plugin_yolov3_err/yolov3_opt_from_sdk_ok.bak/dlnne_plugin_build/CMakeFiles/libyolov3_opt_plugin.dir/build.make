# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build

# Include any dependencies generated for this target.
include CMakeFiles/libyolov3_opt_plugin.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/libyolov3_opt_plugin.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/libyolov3_opt_plugin.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libyolov3_opt_plugin.dir/flags.make

CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o: CMakeFiles/libyolov3_opt_plugin.dir/flags.make
CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o: /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin/plugin.cc
CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o: CMakeFiles/libyolov3_opt_plugin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o -MF CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o.d -o CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o -c /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin/plugin.cc

CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.i"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin/plugin.cc > CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.i

CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.s"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin/plugin.cc -o CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.s

CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o: CMakeFiles/libyolov3_opt_plugin.dir/flags.make
CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o: /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin/utils.cc
CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o: CMakeFiles/libyolov3_opt_plugin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o -MF CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o.d -o CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o -c /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin/utils.cc

CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.i"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin/utils.cc > CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.i

CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.s"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin/utils.cc -o CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.s

CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o: CMakeFiles/libyolov3_opt_plugin.dir/flags.make
CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o: /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin_register.cu
CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o: CMakeFiles/libyolov3_opt_plugin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o -MF CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o.d -o CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o -c /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin_register.cu

CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.i"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin_register.cu > CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.i

CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.s"
	/opt/sdk/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin/plugin_register.cu -o CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.s

# Object files for target libyolov3_opt_plugin
libyolov3_opt_plugin_OBJECTS = \
"CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o" \
"CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o" \
"CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o"

# External object files for target libyolov3_opt_plugin
libyolov3_opt_plugin_EXTERNAL_OBJECTS =

libyolov3_opt_plugin.so: CMakeFiles/libyolov3_opt_plugin.dir/plugin/plugin.cc.o
libyolov3_opt_plugin.so: CMakeFiles/libyolov3_opt_plugin.dir/plugin/utils.cc.o
libyolov3_opt_plugin.so: CMakeFiles/libyolov3_opt_plugin.dir/plugin_register.cu.o
libyolov3_opt_plugin.so: CMakeFiles/libyolov3_opt_plugin.dir/build.make
libyolov3_opt_plugin.so: /opt/sdk/lib/libtvm.so
libyolov3_opt_plugin.so: /opt/sdk/lib/libdlnne.so
libyolov3_opt_plugin.so: /opt/sdk/lib/libdlhc.so
libyolov3_opt_plugin.so: /opt/sdk/lib/libcurt.so
libyolov3_opt_plugin.so: /opt/sdk/lib/libdlhal.so
libyolov3_opt_plugin.so: /opt/sdk/lib/libdli_tu.so
libyolov3_opt_plugin.so: /opt/sdk/lib/libclang-cpp.so
libyolov3_opt_plugin.so: /opt/sdk/lib/libclang.so
libyolov3_opt_plugin.so: CMakeFiles/libyolov3_opt_plugin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libyolov3_opt_plugin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libyolov3_opt_plugin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/libyolov3_opt_plugin.dir/build: libyolov3_opt_plugin.so
.PHONY : CMakeFiles/libyolov3_opt_plugin.dir/build

CMakeFiles/libyolov3_opt_plugin.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libyolov3_opt_plugin.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libyolov3_opt_plugin.dir/clean

CMakeFiles/libyolov3_opt_plugin.dir/depend:
	cd /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build /home/ningzhang/nms_plugin_yolov3_err/yolov3_opt_from_sdk/dlnne_plugin_build/CMakeFiles/libyolov3_opt_plugin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libyolov3_opt_plugin.dir/depend

