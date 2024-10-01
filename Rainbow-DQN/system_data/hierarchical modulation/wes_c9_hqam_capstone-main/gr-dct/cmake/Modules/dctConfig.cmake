INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_DCT dct)

FIND_PATH(
    DCT_INCLUDE_DIRS
    NAMES dct/api.h
    HINTS $ENV{DCT_DIR}/include
        ${PC_DCT_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    DCT_LIBRARIES
    NAMES gnuradio-dct
    HINTS $ENV{DCT_DIR}/lib
        ${PC_DCT_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/dctTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DCT DEFAULT_MSG DCT_LIBRARIES DCT_INCLUDE_DIRS)
MARK_AS_ADVANCED(DCT_LIBRARIES DCT_INCLUDE_DIRS)
