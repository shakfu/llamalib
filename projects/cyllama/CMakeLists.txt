set(PROJECT_NAME cyllama)

set(PROJECT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cyllama.cpp)

add_library( 
    ${PROJECT_NAME} 
    MODULE
    ${PROJECT_SOURCES}
)

add_custom_command(
    OUTPUT ${PROJECT_SOURCES}
    COMMAND cythonize -3 ${PROJECT_NAME}.pyx
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${PROJECT_NAME}.pyx
    COMMENT "Generating ${PROJECT_NAME}.cpp"
)

add_custom_target(
    ${PROJECT_NAME}_cpp
    DEPENDS ${PROJECT_NAME}.pyx
)

add_dependencies(
    ${PROJECT_NAME}
    ${PROJECT_NAME}_cpp
)

string(TOLOWER ${CMAKE_SYSTEM_NAME} platform)
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_NAME "${PROJECT_NAME}.cpython-${Python_VERSION_MAJOR}${Python_VERSION_MINOR}-${platform}"
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${Python_INCLUDE_DIRS}
    ${PROJECT_INCLUDES}
)

target_compile_definitions(
    ${PROJECT_NAME}
    PUBLIC
    -DNDEBUG
)

target_compile_options(
    ${PROJECT_NAME}
    PUBLIC
    -std=c++11 
    $<$<PLATFORM_ID:Darwin>:-Wno-unused-result>
    $<$<PLATFORM_ID:Darwin>:-Wsign-compare>
    $<$<PLATFORM_ID:Darwin>:-Wunreachable-code>
    $<$<PLATFORM_ID:Darwin>:-fno-common>
    $<$<PLATFORM_ID:Darwin>:-Wall>  
    $<$<PLATFORM_ID:Darwin>:-g>
    $<$<PLATFORM_ID:Darwin>:-fwrapv>
    $<$<PLATFORM_ID:Darwin>:-O3>
    $<$<PLATFORM_ID:Windows>:/O2>
    $<$<PLATFORM_ID:Windows>:/MD>
)

target_link_options(
    ${PROJECT_NAME}
    PUBLIC
    $<$<PLATFORM_ID:Darwin>:-dynamic>
)

target_link_directories(
    ${PROJECT_NAME} 
    PUBLIC
    ${Python_LIBRARY_DIRS}
    ${PROJECT_LIB_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME} 
    PUBLIC
    ${Python_LIBRARIES}
    ${LLAMA_LINK_LIBS}
)
