project(PyAggregationTests-FS)

cmake_minimum_required(VERSION 3.10)

# set this to the path to your smartredis installation
set(SMARTREDIS "../../../SmartRedis" CACHE PATH "Path to SmartRedis root directory")

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM
    /usr/local/include
    ${MPI_INCLUDE_PATH}
)

# tests need MPI
find_package(MPI)

add_executable(aggregation_producer
	aggregation_producer.cpp
)
target_link_libraries(aggregation_producer
	MPI::MPI_CXX
)
