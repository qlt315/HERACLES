INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_HQAM hqam)

FIND_PATH(
    HQAM_INCLUDE_DIRS
    NAMES hqam/api.h
    HINTS $ENV{HQAM_DIR}/include
        ${PC_HQAM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    HQAM_LIBRARIES
    NAMES gnuradio-hqam
    HINTS $ENV{HQAM_DIR}/lib
        ${PC_HQAM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/hqamTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(HQAM DEFAULT_MSG HQAM_LIBRARIES HQAM_INCLUDE_DIRS)
MARK_AS_ADVANCED(HQAM_LIBRARIES HQAM_INCLUDE_DIRS)
