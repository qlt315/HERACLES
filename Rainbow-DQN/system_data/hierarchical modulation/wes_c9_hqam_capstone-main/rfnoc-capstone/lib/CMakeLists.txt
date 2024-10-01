# Copyright 2011,2012 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

########################################################################
# Setup library
########################################################################
include(GrPlatform) #define LIB_SUFFIX

list(APPEND capstone_sources
    conv_impl.cc
    conv_block_ctrl_impl.cpp
    conv_impl.cc
    conv_block_ctrl_impl.cpp
)

set(capstone_sources "${capstone_sources}" PARENT_SCOPE)
if(NOT capstone_sources)
	MESSAGE(STATUS "No C++ sources... skipping lib/")
	return()
endif(NOT capstone_sources)

add_library(gnuradio-capstone SHARED ${capstone_sources})
target_link_libraries(gnuradio-capstone ${Boost_LIBRARIES} ${GNURADIO_ALL_LIBRARIES} ${ETTUS_LIBRARIES})
set_target_properties(gnuradio-capstone PROPERTIES DEFINE_SYMBOL "gnuradio_capstone_EXPORTS")

if(APPLE)
    set_target_properties(gnuradio-capstone PROPERTIES
        INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
    )
endif(APPLE)

########################################################################
# Install built library files
########################################################################
install(TARGETS gnuradio-capstone
    LIBRARY DESTINATION lib${LIB_SUFFIX} # .so/.dylib file
    ARCHIVE DESTINATION lib${LIB_SUFFIX} # .lib file
    RUNTIME DESTINATION bin              # .dll file
)

########################################################################
# Build and register unit test
########################################################################
if(CPPUNIT_FOUND)

    include(GrTest)

    include_directories(${CPPUNIT_INCLUDE_DIRS})

    link_directories(${CPPUNIT_LIBRARY_DIRS})

    list(APPEND test_capstone_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/test_capstone.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/qa_capstone.cc
    )

    add_executable(test-capstone ${test_capstone_sources})

    target_link_libraries(
        test-capstone
        ${GNURADIO_RUNTIME_LIBRARIES}
        ${Boost_LIBRARIES}
        ${CPPUNIT_LIBRARIES}
        ${ETTUS_LIBRARIES}
        ${PC_ETTUS_LIBDIR}
        gnuradio-capstone
    )

    GR_ADD_TEST(test_capstone test-capstone)

else(CPPUNIT_FOUND)
    message(WARNING "Disabling unit tests: No CppUnit found.")
endif(CPPUNIT_FOUND)
