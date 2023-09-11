# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

add_library(
  SDFG_UTILS
  Sanitizer.cpp
  NameGenerator.cpp
  IDGenerator.cpp
  GetSizedType.cpp
  GetParents.cpp
  ValueToString.cpp
  AttributeToString.cpp
  OperationToString.cpp)
target_include_directories(SDFG_UTILS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(
  SOURCE_FILES_CPP
  PRIVATE Sanitizer.cpp
          NameGenerator.cpp
          IDGenerator.cpp
          GetSizedType.cpp
          GetParents.cpp
          ValueToString.cpp
          AttributeToString.cpp
          OperationToString.cpp)
