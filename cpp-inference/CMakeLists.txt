project(CppInferenceTests)

set(CMAKE_BUILD_TYPE DEBUG)

cmake_minimum_required(VERSION 3.10)

SET(CMAKE_CXX_STANDARD 17)

# set this to the path to your silc installation
set(SILC ../../silc/)

# find hiredis library and include path
string(CONCAT HIREDIS_LIB_PATH ${SILC} "third-party/hiredis/install/lib")
find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH)
string(CONCAT HIREDIS_INCLUDE_PATH ${SILC} "third-party/hiredis/install/include/")

# find redis++ lib and include path
string(CONCAT REDISPP_LIB_PATH ${SILC} "third-party/redis-plus-plus/install/lib")
find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} NO_DEFAULT_PATH)
string(CONCAT REDISPP_INCLUDE_PATH ${SILC} "third-party/redis-plus-plus/install/include/")

# find protobuf lib and include path
string(CONCAT PROTOBUF_LIB_PATH ${SILC} "third-party/protobuf/install/lib")
find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH)
string(CONCAT PROTOBUF_INCLUDE_PATH ${SILC} "third-party/protobuf/install/include/")

# all SILC dependencies
set(SILC_DEPS ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB})

# SILC internals - requires "make lib" being called in top
# level of the SILC install directory.
string(CONCAT SILC_LIB_PATH ${SILC} "build/")
find_library(SILC_LIB silc PATHS ${SILC_LIB_PATH} NO_DEFAULT_PATH)
string(CONCAT SILC_INCLUDE ${SILC} "include/")
string(CONCAT SILC_PROTO_INCLUDE ${SILC} "utils/protobuf/")

# tests need MPI
find_package(MPI)

include_directories(SYSTEM
    ${MPI_INCLUDE_PATH}
	${SILC_INCLUDE}
	${SILC_PROTO_INCLUDE}
    /usr/local/include
	${HIREDIS_INCLUDE_PATH}
    ${REDISPP_INCLUDE_PATH}
    ${PROTOBUF_INCLUDE_PATH}
)

add_executable(run_mnist_inference
	inference_scaling.cpp
)
target_link_libraries(run_mnist_inference
	MPI::MPI_CXX
	${SILC_DEPS}
	${SILC_LIB}
)

add_executable(run_resnet_inference
	inference_scaling_imagenet.cpp
)
target_link_libraries(run_resnet_inference
	MPI::MPI_CXX
	${SILC_DEPS}
	${SILC_LIB}
)