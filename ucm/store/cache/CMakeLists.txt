file(GLOB_RECURSE UCM_CACHE_STORE_CC_SOURCE_FILES "./cc/*.cc")
file(GLOB_RECURSE UCM_CACHE_STORE_V2_CC_SOURCE_FILES "./v2/*.cc")
# Which cores this platform ships is decided once, in the root CMakeLists, so the
# test suite can be filtered by the same flags.
set(UCM_CACHE_STORE_SOURCES "")
if(UCM_CACHE_STORE_BUILDS_CC)
    list(APPEND UCM_CACHE_STORE_SOURCES ${UCM_CACHE_STORE_CC_SOURCE_FILES})
endif()
if(UCM_CACHE_STORE_BUILDS_V2)
    list(APPEND UCM_CACHE_STORE_SOURCES ${UCM_CACHE_STORE_V2_CC_SOURCE_FILES})
endif()
add_library(cachestore SHARED ${UCM_CACHE_STORE_SOURCES})
if(UCM_CACHE_STORE_BUILDS_CC)
    target_include_directories(cachestore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
endif()
target_link_libraries(cachestore PUBLIC storeintf trans infra_logger metrics)
set_target_properties(cachestore PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../shared/metrics"
)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS cachestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
