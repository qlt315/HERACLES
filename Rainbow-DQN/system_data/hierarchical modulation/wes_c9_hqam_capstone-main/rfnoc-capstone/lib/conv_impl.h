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

#pragma once

#include <capstone/conv.h>
#include <capstone/conv_block_ctrl.hpp>

namespace gr {
  namespace capstone {

    class conv_impl : public conv
    {
     public:
      conv_impl(::uhd::rfnoc::noc_block_base::sptr block_ref);
      ~conv_impl();

      // Where all the action really happens

     private:
      ::uhd::rfnoc::conv_block_ctrl::sptr d_conv_ref;
    };

  } // namespace capstone
} // namespace gr

