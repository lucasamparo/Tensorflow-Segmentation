# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lucasamparo/expression-removal/result/deprojection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lucasamparo/expression-removal/result/deprojection

# Include any dependencies generated for this target.
include CMakeFiles/deproj.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/deproj.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/deproj.dir/flags.make

CMakeFiles/deproj.dir/deprojection.cpp.o: CMakeFiles/deproj.dir/flags.make
CMakeFiles/deproj.dir/deprojection.cpp.o: deprojection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lucasamparo/expression-removal/result/deprojection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/deproj.dir/deprojection.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deproj.dir/deprojection.cpp.o -c /home/lucasamparo/expression-removal/result/deprojection/deprojection.cpp

CMakeFiles/deproj.dir/deprojection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deproj.dir/deprojection.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lucasamparo/expression-removal/result/deprojection/deprojection.cpp > CMakeFiles/deproj.dir/deprojection.cpp.i

CMakeFiles/deproj.dir/deprojection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deproj.dir/deprojection.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lucasamparo/expression-removal/result/deprojection/deprojection.cpp -o CMakeFiles/deproj.dir/deprojection.cpp.s

CMakeFiles/deproj.dir/deprojection.cpp.o.requires:

.PHONY : CMakeFiles/deproj.dir/deprojection.cpp.o.requires

CMakeFiles/deproj.dir/deprojection.cpp.o.provides: CMakeFiles/deproj.dir/deprojection.cpp.o.requires
	$(MAKE) -f CMakeFiles/deproj.dir/build.make CMakeFiles/deproj.dir/deprojection.cpp.o.provides.build
.PHONY : CMakeFiles/deproj.dir/deprojection.cpp.o.provides

CMakeFiles/deproj.dir/deprojection.cpp.o.provides.build: CMakeFiles/deproj.dir/deprojection.cpp.o


# Object files for target deproj
deproj_OBJECTS = \
"CMakeFiles/deproj.dir/deprojection.cpp.o"

# External object files for target deproj
deproj_EXTERNAL_OBJECTS =

deproj: CMakeFiles/deproj.dir/deprojection.cpp.o
deproj: CMakeFiles/deproj.dir/build.make
deproj: /usr/lib/x86_64-linux-gnu/libboost_system.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_thread.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_regex.so
deproj: /usr/lib/x86_64-linux-gnu/libpthread.so
deproj: /usr/local/lib/libpcl_common.so
deproj: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
deproj: /usr/local/lib/libpcl_kdtree.so
deproj: /usr/local/lib/libpcl_octree.so
deproj: /usr/local/lib/libpcl_search.so
deproj: /usr/local/lib/libpcl_sample_consensus.so
deproj: /usr/local/lib/libpcl_filters.so
deproj: /usr/local/lib/libpcl_features.so
deproj: /usr/lib/libOpenNI.so
deproj: /usr/lib/libOpenNI2.so
deproj: /usr/local/lib/libpcl_io.so
deproj: /usr/local/lib/libpcl_segmentation.so
deproj: /usr/local/lib/libpcl_keypoints.so
deproj: /usr/local/lib/libpcl_surface.so
deproj: /usr/local/lib/libpcl_registration.so
deproj: /usr/local/lib/libpcl_recognition.so
deproj: /usr/local/lib/libpcl_tracking.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_system.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_thread.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
deproj: /usr/lib/x86_64-linux-gnu/libboost_regex.so
deproj: /usr/lib/x86_64-linux-gnu/libpthread.so
deproj: /usr/lib/libOpenNI.so
deproj: /usr/lib/libOpenNI2.so
deproj: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
deproj: /usr/local/lib/libopencv_dnn.so.3.3.0
deproj: /usr/local/lib/libopencv_ml.so.3.3.0
deproj: /usr/local/lib/libopencv_objdetect.so.3.3.0
deproj: /usr/local/lib/libopencv_shape.so.3.3.0
deproj: /usr/local/lib/libopencv_stitching.so.3.3.0
deproj: /usr/local/lib/libopencv_superres.so.3.3.0
deproj: /usr/local/lib/libopencv_videostab.so.3.3.0
deproj: /usr/local/lib/libpcl_common.so
deproj: /usr/local/lib/libpcl_kdtree.so
deproj: /usr/local/lib/libpcl_octree.so
deproj: /usr/local/lib/libpcl_search.so
deproj: /usr/local/lib/libpcl_sample_consensus.so
deproj: /usr/local/lib/libpcl_filters.so
deproj: /usr/local/lib/libpcl_features.so
deproj: /usr/local/lib/libpcl_io.so
deproj: /usr/local/lib/libpcl_segmentation.so
deproj: /usr/local/lib/libpcl_keypoints.so
deproj: /usr/local/lib/libpcl_surface.so
deproj: /usr/local/lib/libpcl_registration.so
deproj: /usr/local/lib/libpcl_recognition.so
deproj: /usr/local/lib/libpcl_tracking.so
deproj: /usr/local/lib/libopencv_calib3d.so.3.3.0
deproj: /usr/local/lib/libopencv_features2d.so.3.3.0
deproj: /usr/local/lib/libopencv_flann.so.3.3.0
deproj: /usr/local/lib/libopencv_highgui.so.3.3.0
deproj: /usr/local/lib/libopencv_photo.so.3.3.0
deproj: /usr/local/lib/libopencv_video.so.3.3.0
deproj: /usr/local/lib/libopencv_videoio.so.3.3.0
deproj: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
deproj: /usr/local/lib/libopencv_imgproc.so.3.3.0
deproj: /usr/local/lib/libopencv_core.so.3.3.0
deproj: CMakeFiles/deproj.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lucasamparo/expression-removal/result/deprojection/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable deproj"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/deproj.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/deproj.dir/build: deproj

.PHONY : CMakeFiles/deproj.dir/build

CMakeFiles/deproj.dir/requires: CMakeFiles/deproj.dir/deprojection.cpp.o.requires

.PHONY : CMakeFiles/deproj.dir/requires

CMakeFiles/deproj.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/deproj.dir/cmake_clean.cmake
.PHONY : CMakeFiles/deproj.dir/clean

CMakeFiles/deproj.dir/depend:
	cd /home/lucasamparo/expression-removal/result/deprojection && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lucasamparo/expression-removal/result/deprojection /home/lucasamparo/expression-removal/result/deprojection /home/lucasamparo/expression-removal/result/deprojection /home/lucasamparo/expression-removal/result/deprojection /home/lucasamparo/expression-removal/result/deprojection/CMakeFiles/deproj.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/deproj.dir/depend

