INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_DD_PLL dd_pll)

FIND_PATH(
    DD_PLL_INCLUDE_DIRS
    NAMES dd_pll/api.h
    HINTS $ENV{DD_PLL_DIR}/include
        ${PC_DD_PLL_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    DD_PLL_LIBRARIES
    NAMES gnuradio-dd_pll
    HINTS $ENV{DD_PLL_DIR}/lib
        ${PC_DD_PLL_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/dd_pllTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DD_PLL DEFAULT_MSG DD_PLL_LIBRARIES DD_PLL_INCLUDE_DIRS)
MARK_AS_ADVANCED(DD_PLL_LIBRARIES DD_PLL_INCLUDE_DIRS)
