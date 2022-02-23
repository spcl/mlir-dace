add_library (
  SDFG_UTILS 
  Sanitizer.cpp
  NameGenerator.cpp
  IDGenerator.cpp
  GetSizedType.cpp
  GetParents.cpp
)
target_include_directories (SDFG_UTILS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
