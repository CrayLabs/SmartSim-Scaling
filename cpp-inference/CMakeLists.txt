project(CppInferenceTests)

set(CMAKE_BUILD_TYPE DEBUG)

cmake_minimum_required(VERSION 3.10)

SET(CMAKE_CXX_STANDARD 17)

# set this to the path to your smartredis installation
set(SMARTREDIS ../smartredis-0.1.0/)

# find hiredis library and include path
string(CONCAT HIREDIS_LIB_PATH ${SMARTREDIS} "third-party/hiredis/install/lib")
find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH)
string(CONCAT HIREDIS_INCLUDE_PATH ${SMARTREDIS} "third-party/hiredis/install/include/")

# find redis++ lib and include path
string(CONCAT REDISPP_LIB_PATH ${SMARTREDIS} "third-party/redis-plus-plus/install/lib")
find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} NO_DEFAULT_PATH)
string(CONCAT REDISPP_INCLUDE_PATH ${SMARTREDIS} "third-party/redis-plus-plus/install/include/")

# find protobuf lib and include path
string(CONCAT PROTOBUF_LIB_PATH ${SMARTREDIS} "third-party/protobuf/install/lib")
find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH)
string(CONCAT PROTOBUF_INCLUDE_PATH ${SMARTREDIS} "third-party/protobuf/install/include/")

# all SmartRedis dependencies
set(SMARTREDIS_DEPS ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB})

# SmartRedis internals - requires "make lib" being called in top
# level of the SmartRedis install directory.
string(CONCAT SR_LIB_PATH ${SMARTREDIS} "build/")
find_library(SR_LIB smartredis PATHS ${SR_LIB_PATH} NO_DEFAULT_PATH)
string(CONCAT SR_INCLUDE ${SMARTREDIS} "include/")
string(CONCAT SR_PROTO_INCLUDE ${SMARTREDIS} "utils/protobuf/")

# tests need MPI
find_package(MPI)

include_directories(SYSTEM
    ${MPI_INCLUDE_PATH}
	${SR_INCLUDE}
	${SR_PROTO_INCLUDE}
    /usr/local/include
	${HIREDIS_INCLUDE_PATH}
    ${REDISPP_INCLUDE_PATH}
    ${PROTOBUF_INCLUDE_PATH}
)

add_executable(run_resnet_inference
	inference_scaling_imagenet.cpp
)
target_link_libraries(run_resnet_inference
	MPI::MPI_CXX
	${SMARTREDIS_DEPS}
	${SR_LIB}
)
