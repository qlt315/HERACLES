/* -*- c++ -*- */
/*
 * Copyright 2022 gr-dd_pll author.
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

#ifndef INCLUDED_DD_PLL_QAM_PLL_IMPL_H
#define INCLUDED_DD_PLL_QAM_PLL_IMPL_H

#include <dd_pll/qam_pll.h>

namespace gr {
  namespace dd_pll {

    class qam_pll_impl : public qam_pll
    {
     private:
      float d_zeta;
      float d_fn;
      double d_sample_freq;
      float f_in;
      float f_in_last;
      float f_int_out;
      float f_out_last;
      float f_out;
      float f_int_out_last;
      float vco_in_last;
      float vco_out;
      float vco_in;
      float vco_out_last;
      
      gr_complex call_pll(const gr_complex &sample);

     public:
      qam_pll_impl(float zeta, float fn, double sample_freq);
      ~qam_pll_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace dd_pll
} // namespace gr

#endif /* INCLUDED_DD_PLL_QAM_PLL_IMPL_H */

