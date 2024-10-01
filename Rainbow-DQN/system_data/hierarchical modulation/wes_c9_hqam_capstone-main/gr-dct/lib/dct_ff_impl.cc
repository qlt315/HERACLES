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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "dct_ff_impl.h"

/* [JC] Uncomment any of these definitions to enable debug printouts */
//#define DEBUG0
//#define DEBUG1
//#define DEBUG2
//#define DEBUG3


namespace gr {
  namespace dct {

    dct_ff::sptr
    dct_ff::make(int width, int height, dct_mode_t mode)
    {
      return gnuradio::get_initial_sptr
        (new dct_ff_impl(width, height, mode));
    }


    /*
     * The private constructor
     */
    dct_ff_impl::dct_ff_impl(int width, int height, dct_mode_t mode)
      : gr::block("dct_ff",
              gr::io_signature::make(1, 1, sizeof(float)),
              gr::io_signature::make(1, 1, sizeof(float)))
    {
        /* [JC] Assign n_cols and n_rows parameter values provided via GRC */        
        n_cols = width;
        n_rows = height;
		  
		/* [JC] Allocate memory for total image FIFO */
		//img_fifo = new float[n_cols * n_rows];
		  
		/* [JC] This keeps track of total image size being processed in samples */
		num_imgsamps = n_cols * n_rows;
		  
		/* [JC] Ensure blocks coming in are multiples of n_cols * n_rows */
		set_output_multiple(num_imgsamps);
        
        /* [JC] DCT modes:  
                1) Forward DCT (default):   dct_mode = 0
                2) Inverse DCT:             dct_mode = 1 
        */
        dct_mode = mode;
    }

    /*
     * Our virtual destructor.
     */
    dct_ff_impl::~dct_ff_impl()
    {
    }

    void
    dct_ff_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      /* <+forecast+> e.g. ninput_items_required[0] = noutput_items */
           
      ninput_items_required[0] = noutput_items;
    }

    int
    dct_ff_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const float *in = (const float *) input_items[0];
      float *out = (float *) output_items[0];
      
    #ifdef DEBUG0
      std::cout << "\n*** noutput_items = " << noutput_items << std::endl;
    #endif		

	  /* [JC] cv::Mat input and output frame matrices */
	  cv::Mat in_frame, out_frame;
	  in_frame.create(n_rows, n_cols, CV_32FC1); 
	  out_frame.create(n_rows, n_cols, CV_32FC1);	
	  //cv::Mat in_frame = cv::Mat(n_rows, n_cols, CV_32FC1, const_cast<float *>(in));	 
	  //cv::Mat out_frame = cv::Mat(n_rows, n_cols, CV_32FC1, out);  			  
		
	  for (int i = 0; i < noutput_items; i++)
		  in_frame.at<float>(i) = in[i] ;

	  switch (dct_mode) {     
		  /* dct_mode == 0: Forward DCT mode */
		  case FWD_DCT: 
			#ifdef DEBUG1
			  std::cout << "We are in Forward DCT mode!" << std::endl;  
			#endif            
 
              /* [JC] Assumption here is that image is already gray scale (1 channel) 
			   *      If it is desired to process color images, gray scale conversion
			   *      must be performed first.
			   */	  
			  
            #ifdef DEBUG2			  
			  for (int i = 0; i < 10; i++)
				  std::cout << "*** in_frame[i] == " << in_frame.at<float>(i) 
				            << std::endl;
            #endif			  
			  
			  /* [JC] Ensure input image is normalized prior to DCT */			  
			  in_frame.convertTo(in_frame, CV_32FC1, 1.0 / 255.0);
			  
            #ifdef DEBUG2
			  for (int i = 0; i < 10; i++)
				  std::cout << "*** Normalized in_frame[i] == " << in_frame.at<float>(i) 
				            << std::endl;			  
            #endif
			  
			  /* [JC] Perform OpenCV DCT on image and scale back */
			  cv::dct(in_frame, out_frame);
			  out_frame.convertTo(out_frame, CV_32FC1, 1.0);     
			  
            #ifdef DEBUG2			  
			  for (int i = 0; i < 10; i++)
				  std::cout << "*** DCT out_frame[i] == " << out_frame.at<float>(i) 
				            << std::endl;				  
            #endif			  

			  break;
		  /* dct_mode == 1: Reverse DCT mode */
		  case REV_DCT:  
			#ifdef DEBUG1     
			  std::cout << "We are in Reverse DCT (IDCT) mode!" << std::endl;
			#endif

            #ifdef DEBUG2				  
			  for (int i = 0; i < 10; i++)
				  std::cout << "+++ Pre-IDCT: in_frame[i] == " << in_frame.at<float>(i) 
				            << std::endl;				  
			#endif
			  
			  /* [JC] Ensure input image is 32-bit float prior to IDCT */			  
			  in_frame.convertTo(in_frame, CV_32FC1, 1.0);
			  
            #ifdef DEBUG3				  
			  for (int i = 0; i < 10; i++)
				  std::cout << "+++ Pre-IDCT: 1/255-scaled in_frame[i] == " 
				            << in_frame.at<float>(i) << std::endl;					  
            #endif
			  
			  /* [JC] Perform OpenCV IDCT on image and convert to byte stream */
			  cv::idct(in_frame, out_frame);
			  
            #ifdef DEBUG2				  
			  for (int i = 0; i < 10; i++)
				  std::cout << "+++ Post-IDCT out_frame[i] == " 
				            << out_frame.at<float>(i) << std::endl;				  
			#endif
			  
			  out_frame.convertTo(out_frame, CV_32FC1, 255.0);  
			  
			#ifdef DEBUG3
			  for (int i = 0; i < 10; i++)
				  std::cout << "+++ Post-IDCT 32-bit 255-scaled out_frame[i] == " 
				            << out_frame.at<float>(i) << std::endl << std::endl;					  
            #endif
			  
			  break;
	  }
		
	  //memcpy(out, out_frame., sizeof(gr_complex) * noutput_items);
	  for (int i = 0; i < noutput_items; i++)
		  out[i] = out_frame.at<float>(i);

      
      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (noutput_items);		
    
      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace dct */
} /* namespace gr */

