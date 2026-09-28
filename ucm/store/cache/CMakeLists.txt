file(GLOB_RECURSE UCM_CACHE_STORE_CC_SOURCE_FILES "./cc/*.cc")
file(GLOB_RECURSE UCM_CACHE_STORE_V2_CC_SOURCE_FILES "./v2/*.cc")
# cache.v2 is the single core for the ascend family and cuda/simu; musa/maca keep
# the cc core. The two must never land in one library: both export MakeCacheStore.
if(RUNTIME_ENVIRONMENT STREQUAL "ascend" OR RUNTIME_ENVIRONMENT STREQUAL "ascend-a3" OR
   RUNTIME_ENVIRONMENT STREQUAL "ascend-a5" OR RUNTIME_ENVIRONMENT STREQUAL "cuda" OR
   RUNTIME_ENVIRONMENT STREQUAL "simu")
    add_library(cachestore SHARED ${UCM_CACHE_STORE_V2_CC_SOURCE_FILES})
else()
    add_library(cachestore SHARED ${UCM_CACHE_STORE_CC_SOURCE_FILES})
    target_include_directories(cachestore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
endif()
target_link_libraries(cachestore PUBLIC storeintf trans infra_logger metrics)
set_target_properties(cachestore PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../shared/metrics"
)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS cachestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
