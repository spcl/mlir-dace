add_library (
  SDIR_UTILS 
  Sanitizer.cpp
  NameGenerator.cpp
  IDGenerator.cpp
)
target_include_directories (SDIR_UTILS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
