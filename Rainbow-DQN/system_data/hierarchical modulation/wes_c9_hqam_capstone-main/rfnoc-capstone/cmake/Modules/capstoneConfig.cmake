INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_CAPSTONE capstone)

FIND_PATH(
    CAPSTONE_INCLUDE_DIRS
    NAMES capstone/api.h
    HINTS $ENV{CAPSTONE_DIR}/include
        ${PC_CAPSTONE_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    CAPSTONE_LIBRARIES
    NAMES gnuradio-capstone
    HINTS $ENV{CAPSTONE_DIR}/lib
        ${PC_CAPSTONE_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CAPSTONE DEFAULT_MSG CAPSTONE_LIBRARIES CAPSTONE_INCLUDE_DIRS)
MARK_AS_ADVANCED(CAPSTONE_LIBRARIES CAPSTONE_INCLUDE_DIRS)

