file(GLOB_RECURSE UCM_CACHE_STORE_CC_SOURCE_FILES "./cc/*.cc")
file(GLOB_RECURSE UCM_CACHE_STORE_V2_CC_SOURCE_FILES "./v2/*.cc")
# A deploy platform builds exactly one core. simu has to carry both: the shared
# ucmstore.test binary globs every case/*.cc, so it needs the cc and v2 suites at
# once, and a single library can only export one MakeCacheStore (see
# v2/cache_store.cc for how the clash is handled).
if(RUNTIME_ENVIRONMENT STREQUAL "ascend-a5")
    add_library(cachestore SHARED ${UCM_CACHE_STORE_V2_CC_SOURCE_FILES})
elseif(RUNTIME_ENVIRONMENT STREQUAL "simu")
    add_library(cachestore SHARED ${UCM_CACHE_STORE_CC_SOURCE_FILES} ${UCM_CACHE_STORE_V2_CC_SOURCE_FILES})
    target_include_directories(cachestore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
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
