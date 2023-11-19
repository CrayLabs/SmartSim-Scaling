project(FortranInferenceTests)

cmake_minimum_required(VERSION 3.10)

enable_language(Fortran)
# set this to the path to your smartredis installation
set(SMARTREDIS "../../SmartRedis" CACHE PATH "Path to SmartRedis root directory")

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE RELEASE)

find_library(SR_LIB smartredis PATHS ${SMARTREDIS}/install/lib NO_DEFAULT_PATH REQUIRED)
find_library(SR_LIB_FORTRAN smartredis-fortran PATHS ${SMARTREDIS}/install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ${MPI_INCLUDE_PATH}
    ${SMARTREDIS}/install/include
)

# tests need MPI
find_package(MPI)

add_executable(run_resnet_inference
	inference_scaling_imagenet.F90
	utils.F90
)
target_link_libraries(run_resnet_inference
	#link the fortran inference against the following libraries
	MPI::MPI_CXX #might need to turn this to the fortran lib MPI::MPI_Fortran
	${SR_LIB}
	${SR_LIB_FORTRAN}
)
