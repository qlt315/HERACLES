INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_CHECKEVM checkevm)

FIND_PATH(
    CHECKEVM_INCLUDE_DIRS
    NAMES checkevm/api.h
    HINTS $ENV{CHECKEVM_DIR}/include
        ${PC_CHECKEVM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    CHECKEVM_LIBRARIES
    NAMES gnuradio-checkevm
    HINTS $ENV{CHECKEVM_DIR}/lib
        ${PC_CHECKEVM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/checkevmTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CHECKEVM DEFAULT_MSG CHECKEVM_LIBRARIES CHECKEVM_INCLUDE_DIRS)
MARK_AS_ADVANCED(CHECKEVM_LIBRARIES CHECKEVM_INCLUDE_DIRS)
