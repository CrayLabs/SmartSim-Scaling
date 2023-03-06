project(PyAggregationTests)

cmake_minimum_required(VERSION 3.10)

# set this to the path to your smartredis installation
set(SMARTREDIS "../../SmartRedis" CACHE PATH "Path to SmartRedis root directory")

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)

find_library(SR_LIB smartredis PATHS ${SMARTREDIS}/install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ${MPI_INCLUDE_PATH}
    ${SMARTREDIS}/install/include
)

# tests need MPI
find_package(MPI)

add_executable(aggregation_producer
	 aggregation_producer.cpp
)
target_link_libraries(aggregation_producer
	MPI::MPI_CXX
	${SR_LIB}
)
