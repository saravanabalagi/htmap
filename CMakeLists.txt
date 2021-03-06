cmake_minimum_required(VERSION 2.8.3)
project(htmap)

# Catkin dependencies
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    image_transport
    cv_bridge
    geometry_msgs
    obindex
    )

# System dependencies
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Defining the package
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES keypoint_description_htmap #htmap
    CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    image_transport
    cv_bridge
    geometry_msgs
    obindex
    DEPENDS OpenCV Boost
)

###########
## Build ##
###########

# Including directories.
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

# Description Library
add_library(description_htmap
    src/imgdesc/KeypointDetector.cpp
    src/imgdesc/KeypointDescriptor.cpp
    src/imgdesc/GlobalDescriptor.cpp
    src/imgdesc/ldb.cpp)
target_link_libraries(description_htmap ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

# HTMap Library
set(HTMAP_SRCS
    src/bayes/BayesFilter.cpp
    src/util/Params.cpp
    src/util/Statistics.cpp
    src/util/Util.cpp
    src/map/Location.cpp
    src/map/HighLevelMap.cpp
    src/lc/LoopCloser.cpp
    src/HTMap.cpp
)
add_library(htmap ${HTMAP_SRCS})
target_link_libraries(htmap description_htmap ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

# HTMap Node
add_executable(htmap_node src/nodes/HTMap_node.cpp)
target_link_libraries(htmap_node htmap)
