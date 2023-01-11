cmake_minimum_required(VERSION 3.13.4)
project(sdfg-opt LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_custom_target(SOURCE_FILES_CPP DEPENDS print-source-files-cpp)
add_custom_target(SOURCE_FILES_H DEPENDS print-source-files-h)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(sdfg-opt)
add_subdirectory(sdfg-translate)

# ##############################################################################
# Formatting & Static Analysis
# ##############################################################################

get_target_property(SOURCE_FILES_CPP SOURCE_FILES_CPP SOURCES)
get_target_property(SOURCE_FILES_H SOURCE_FILES_H SOURCES)
set(SOURCE_FILES ${SOURCE_FILES_CPP} ${SOURCE_FILES_H})

add_custom_command(
  OUTPUT print-source-files-cpp
  COMMENT "Printing all source files (cpp) considered in static checks"
  COMMAND echo ${SOURCE_FILES_CPP} | tr \" \" \"\\n\")

add_custom_command(
  OUTPUT print-source-files-h
  COMMENT "Printing all source files (h) considered in static checks"
  COMMAND echo ${SOURCE_FILES_H} | tr \" \" \"\\n\")

find_program(CLANG_FORMAT NAMES clang-format)

if(CLANG_FORMAT)
  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT} -i -style=file ${SOURCE_FILES}
    COMMENT "Auto formatting all source files")

  add_custom_target(
    check-format
    COMMAND
      ${CLANG_FORMAT} --style=file --output-replacements-xml ${SOURCE_FILES} |
      tee ${CMAKE_BINARY_DIR}/check_format_file.txt | grep -c "replacement " |
      tr -d "[:cntrl:]" && echo " replacements necessary"
    COMMAND ! grep -c "replacement " ${CMAKE_BINARY_DIR}/check_format_file.txt >
            /dev/null
    COMMENT "Checking format compliance")
else()
  message(STATUS "Could NOT find clang-format")
endif()
