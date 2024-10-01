#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gnuradio::gnuradio-coarsefreq" for configuration "Release"
set_property(TARGET gnuradio::gnuradio-coarsefreq APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gnuradio::gnuradio-coarsefreq PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgnuradio-coarsefreq.so.1.0.0.0"
  IMPORTED_SONAME_RELEASE "libgnuradio-coarsefreq.so.1.0.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS gnuradio::gnuradio-coarsefreq )
list(APPEND _IMPORT_CHECK_FILES_FOR_gnuradio::gnuradio-coarsefreq "${_IMPORT_PREFIX}/lib/libgnuradio-coarsefreq.so.1.0.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
