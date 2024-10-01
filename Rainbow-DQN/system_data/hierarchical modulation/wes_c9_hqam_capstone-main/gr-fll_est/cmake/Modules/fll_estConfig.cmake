INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_FLL_EST fll_est)

FIND_PATH(
    FLL_EST_INCLUDE_DIRS
    NAMES fll_est/api.h
    HINTS $ENV{FLL_EST_DIR}/include
        ${PC_FLL_EST_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    FLL_EST_LIBRARIES
    NAMES gnuradio-fll_est
    HINTS $ENV{FLL_EST_DIR}/lib
        ${PC_FLL_EST_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/fll_estTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FLL_EST DEFAULT_MSG FLL_EST_LIBRARIES FLL_EST_INCLUDE_DIRS)
MARK_AS_ADVANCED(FLL_EST_LIBRARIES FLL_EST_INCLUDE_DIRS)
