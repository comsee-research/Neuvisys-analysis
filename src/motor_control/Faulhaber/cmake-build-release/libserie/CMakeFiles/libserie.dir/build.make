# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /home/thomas/Apps/Clion/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/thomas/Apps/Clion/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release

# Include any dependencies generated for this target.
include libserie/CMakeFiles/libserie.dir/depend.make
# Include the progress variables for this target.
include libserie/CMakeFiles/libserie.dir/progress.make

# Include the compile flags for this target's objects.
include libserie/CMakeFiles/libserie.dir/flags.make

libserie/CMakeFiles/libserie.dir/comserie.cpp.o: libserie/CMakeFiles/libserie.dir/flags.make
libserie/CMakeFiles/libserie.dir/comserie.cpp.o: ../libserie/comserie.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libserie/CMakeFiles/libserie.dir/comserie.cpp.o"
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libserie.dir/comserie.cpp.o -c /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/libserie/comserie.cpp

libserie/CMakeFiles/libserie.dir/comserie.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libserie.dir/comserie.cpp.i"
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/libserie/comserie.cpp > CMakeFiles/libserie.dir/comserie.cpp.i

libserie/CMakeFiles/libserie.dir/comserie.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libserie.dir/comserie.cpp.s"
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/libserie/comserie.cpp -o CMakeFiles/libserie.dir/comserie.cpp.s

libserie/CMakeFiles/libserie.dir/Faulhaber.cpp.o: libserie/CMakeFiles/libserie.dir/flags.make
libserie/CMakeFiles/libserie.dir/Faulhaber.cpp.o: ../libserie/Faulhaber.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object libserie/CMakeFiles/libserie.dir/Faulhaber.cpp.o"
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libserie.dir/Faulhaber.cpp.o -c /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/libserie/Faulhaber.cpp

libserie/CMakeFiles/libserie.dir/Faulhaber.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libserie.dir/Faulhaber.cpp.i"
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/libserie/Faulhaber.cpp > CMakeFiles/libserie.dir/Faulhaber.cpp.i

libserie/CMakeFiles/libserie.dir/Faulhaber.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libserie.dir/Faulhaber.cpp.s"
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/libserie/Faulhaber.cpp -o CMakeFiles/libserie.dir/Faulhaber.cpp.s

# Object files for target libserie
libserie_OBJECTS = \
"CMakeFiles/libserie.dir/comserie.cpp.o" \
"CMakeFiles/libserie.dir/Faulhaber.cpp.o"

# External object files for target libserie
libserie_EXTERNAL_OBJECTS =

libserie/liblibserie.a: libserie/CMakeFiles/libserie.dir/comserie.cpp.o
libserie/liblibserie.a: libserie/CMakeFiles/libserie.dir/Faulhaber.cpp.o
libserie/liblibserie.a: libserie/CMakeFiles/libserie.dir/build.make
libserie/liblibserie.a: libserie/CMakeFiles/libserie.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library liblibserie.a"
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && $(CMAKE_COMMAND) -P CMakeFiles/libserie.dir/cmake_clean_target.cmake
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libserie.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libserie/CMakeFiles/libserie.dir/build: libserie/liblibserie.a
.PHONY : libserie/CMakeFiles/libserie.dir/build

libserie/CMakeFiles/libserie.dir/clean:
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie && $(CMAKE_COMMAND) -P CMakeFiles/libserie.dir/cmake_clean.cmake
.PHONY : libserie/CMakeFiles/libserie.dir/clean

libserie/CMakeFiles/libserie.dir/depend:
	cd /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/libserie /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie /home/thomas/neuvisys-analysis/src/motor_control/Faulhaber/cmake-build-release/libserie/CMakeFiles/libserie.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libserie/CMakeFiles/libserie.dir/depend

