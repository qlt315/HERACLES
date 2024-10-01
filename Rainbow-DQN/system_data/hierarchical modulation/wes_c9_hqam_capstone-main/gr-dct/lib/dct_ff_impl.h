/* -*- c++ -*- */
/*
 * Copyright 2022 Jeff Cuenco.
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

#ifndef INCLUDED_DCT_DCT_FF_IMPL_H
#define INCLUDED_DCT_DCT_FF_IMPL_H

#include <dct/dct_ff.h>

/* [JC] 21May2022 - Add OpenCV2 headers */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace gr {
  namespace dct {
  
    class dct_ff_impl : public dct_ff
    {
     private:
      /* [JC] Image frame width and height and sample size (calculated from n_cols * n_rows) */
      int n_cols;
      int n_rows;
	  int num_imgsamps;
      
      /* [JC] DCT mode */
      dct_mode_t dct_mode;        	 

     public:
      dct_ff_impl(int width, int height, dct_mode_t mode);
      ~dct_ff_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace dct
} // namespace gr

#endif /* INCLUDED_DCT_DCT_FF_IMPL_H */

