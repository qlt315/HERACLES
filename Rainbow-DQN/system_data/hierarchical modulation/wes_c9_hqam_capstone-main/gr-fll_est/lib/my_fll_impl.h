/* -*- c++ -*- */
/*
 * Copyright 2022 gr-fll_est author.
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

#ifndef INCLUDED_FLL_EST_MY_FLL_IMPL_H
#define INCLUDED_FLL_EST_MY_FLL_IMPL_H

#include <fll_est/my_fll.h>
#include <gnuradio/filter/fft_filter.h>


using namespace gr::filter;

namespace gr {
  namespace fll_est {

    class my_fll_impl : public my_fll
    {
     private:
      // Nothing to declare in this block.
      pmt::pmt_t d_src_id;
      std::vector<gr_complex> d_symbols;
      float d_sps;
      unsigned int d_mark_delay, d_stashed_mark_delay;
      float d_thresh, d_stashed_threshold;
      kernel::fft_filter_ccc* d_filter;
      float c_phase;

      gr_complex* d_corr;
      float* d_corr_mag;

      float d_scale;
      float d_pfa; // probability of false alarm

      tm_type d_threshold_method;

      void _set_mark_delay(unsigned int mark_delay);
      void _set_threshold(float threshold);

     public:
      my_fll_impl(const std::vector<gr_complex>& symbols,
                     float sps,
                     unsigned int mark_delay,
                     float threshold = 0.9,
                     tm_type threshold_method = THRESHOLD_ABSOLUTE);
      ~my_fll_impl();
      
      std::vector<gr_complex> symbols() const;
      void set_symbols(const std::vector<gr_complex>& symbols);

      unsigned int mark_delay() const;
      void set_mark_delay(unsigned int mark_delay);

      float threshold() const;
      void set_threshold(float threshold);

      // Where all the action really happens
      int work(
              int noutput_items,
              gr_vector_const_void_star &input_items,
              gr_vector_void_star &output_items
      );
    };

  } // namespace fll_est
} // namespace gr

#endif /* INCLUDED_FLL_EST_MY_FLL_IMPL_H */

