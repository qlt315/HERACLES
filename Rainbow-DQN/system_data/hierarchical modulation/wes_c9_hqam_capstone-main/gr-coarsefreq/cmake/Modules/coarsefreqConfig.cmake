INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_COARSEFREQ coarsefreq)

FIND_PATH(
    COARSEFREQ_INCLUDE_DIRS
    NAMES coarsefreq/api.h
    HINTS $ENV{COARSEFREQ_DIR}/include
        ${PC_COARSEFREQ_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    COARSEFREQ_LIBRARIES
    NAMES gnuradio-coarsefreq
    HINTS $ENV{COARSEFREQ_DIR}/lib
        ${PC_COARSEFREQ_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/coarsefreqTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(COARSEFREQ DEFAULT_MSG COARSEFREQ_LIBRARIES COARSEFREQ_INCLUDE_DIRS)
MARK_AS_ADVANCED(COARSEFREQ_LIBRARIES COARSEFREQ_INCLUDE_DIRS)
