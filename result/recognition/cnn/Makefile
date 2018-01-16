CXX = g++

CXXFLAGS = -std=c++14 -O3 -ffast-math -D CPU_ONLY -D EIGEN_DONT_PARALLELIZE

LDFLAGS = -lpthread -lz `pkg-config --libs opencv` -L ~/Git/caffe/build/lib -lboost_system -lcaffe -lglog -lpthread -lprotobuf

INCLUDE = `pkg-config --cflags opencv` -I /usr/local/include/liblzf/ -I $(CURDIR)/include -I ~/Git/caffe/include

all: record_video.out load.out extract_descriptors.out calculate_psafe.out

psafe: calculate_psafe.out

build/realsense.o: src/realsense.cpp include/realsense.hpp include/camera.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

build/loadrealsense.o: src/loadrealsense.cpp include/loadrealsense.hpp include/camera.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

build/capture.o: src/capture.cpp include/capture.hpp include/camera.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

build/detector.o: src/detector.cpp include/detector.hpp include/util.hpp include/camera.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

build/record_video.o: src/record_video.cpp include/realsense.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

build/load.o: src/load.cpp include/camera.hpp include/aligner.hpp include/loadrealsense.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

build/%.o: src/%.cpp include/%.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

build/extract_descriptors.o: src/extract_descriptors.cpp include/descriptor.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

extract_descriptors.out: build/extract_descriptors.o build/detector.o build/util.o build/normalize.o build/descriptor.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

record_video.out: build/record_video.o build/realsense.o build/capture.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

load.out: build/loadrealsense.o build/load.o build/aligner.o build/detector.o build/util.o build/normalize.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/calculate_psafe.o: src/calculate_psafe.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

calculate_psafe.out: build/calculate_psafe.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	
clean:
	@rm -f build/*.o *.out
