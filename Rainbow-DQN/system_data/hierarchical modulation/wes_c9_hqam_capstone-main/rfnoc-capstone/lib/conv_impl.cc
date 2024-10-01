/* -*- c++ -*- */
/* 
 * Copyright 2022 <+YOU OR YOUR COMPANY+>.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "conv_impl.h"
namespace gr {
  namespace capstone {
    conv::sptr
    conv::make(
        const gr::ettus::rfnoc_graph::sptr graph,
        const ::uhd::device_addr_t& block_args,
        const int device_select,
        const int instance
    )
    {
      return gnuradio::get_initial_sptr(
        new conv_impl(
          gr::ettus::rfnoc_block::make_block_ref(
            graph, block_args, "conv", device_select, instance)));
    }

    /*
     * The private constructor
     */
    conv_impl::conv_impl(
        ::uhd::rfnoc::noc_block_base::sptr block_ref) :
      gr::ettus::rfnoc_block(block_ref),
      d_conv_ref(get_block_ref<::uhd::rfnoc::conv_block_ctrl>())
    {
    }

    /*
     * Our virtual destructor.
     */
    conv_impl::~conv_impl()
    {
    }

  } /* namespace capstone */
} /* namespace gr */

