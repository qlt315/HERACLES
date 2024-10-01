/* -*- c++ -*- */
/*
 * Copyright 2022 gr-coarsefreq author.
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
#include "c_freq_offset_impl.h"
#include <gnuradio/expj.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/math.h>
#include <gnuradio/sincos.h>
#include <boost/format.hpp>
#include <gnuradio/blocks/complex_to_arg.h>
#include <gnuradio/blocks/conjugate_cc.h>
#include <cmath>
#include <iostream>

namespace gr {
  namespace coarsefreq {

    c_freq_offset::sptr
    c_freq_offset::make(int packet_len, float threshold, double sample_freq)
    {
      return gnuradio::get_initial_sptr
        (new c_freq_offset_impl(packet_len, threshold, sample_freq));
    }


    /*
     * The private constructor
     */
    c_freq_offset_impl::c_freq_offset_impl(int packet_len, float threshold, double sample_freq)
      : gr::block("c_freq_offset",
              gr::io_signature::make(2, 2, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
              d_packet_len(packet_len),
              d_threshold(threshold),
              d_sample_freq(sample_freq),
              theta(0),
              incr_count(0)
    {}

    /*
     * Our virtual destructor.
     */
    c_freq_offset_impl::~c_freq_offset_impl()
    {
    }

    void
    c_freq_offset_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      unsigned ninputs = ninput_items_required.size ();
      
      //cout << "noutput_items forecast -- " <<noutput_items << endl;
      for(unsigned i = 0; i < ninputs; i++)
      	ninput_items_required[i] = noutput_items; 
    }
    
    // my function to detect the peaks
/*    gr_complex c_freq_offset_impl::frame_sync(const gr_complex &sample, const gr_complex &corr)
    {
    	float mag_corr;
    	
    	mag_corr = std::abs(corr);
    	
    	if (mag_corr > d_threshold)
    	{
    		
    	}
    	
    	
    
    
    }

*/
    
    /* we are taking two inputs in c_freq_offset_impl block, 1. the received sample and 2. the correlated output of y(n) & y(n+64). 
    The correlated output should yield a plateau since we are repeating our 32 symbol(for QPSK or 64 symbol for BPSK) twice to produce
    a 64 symbol preamble, which lets us take auto correlation of received signal to calculate phase/frequency offset */

    int
    c_freq_offset_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex *sig_in = (const gr_complex *) input_items[0];
      const gr_complex *corr_in = (const gr_complex *) input_items[1];
      gr_complex *out = (gr_complex *) output_items[0];
      int n_samples_afterthreshold = 0;
      

      // Do <+signal processing+>
      for(int i=0; i < noutput_items; i++)
      {

	if (corr_in[i].real() >= d_threshold)
	{
		//printf("threshold met \n");
		n_samples_afterthreshold += 1;
		if (n_samples_afterthreshold == 20)
		{
			
			//if (incr_count == 0)
			//{
				//printf("should be printed after 15 samples \n");
				theta = std::arg(corr_in[i]);
				
				printf("theta is %f at sample %ld \n ", theta,incr_count+i);
			//}
			
		}
	}
	else
	{
		n_samples_afterthreshold = 0;
	} 
	
	out[i] = sig_in[i]*gr_expj(-theta);
	
	
	
	//printf("Input Sample %f + i %f and output Sample %f + i %f", sig_in[i].real(), sig_in[i].imag(),out[i].real(), out[i].imag());
	
		
	      	
      }
      
      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (noutput_items);
      
      incr_count += noutput_items;
      
      /*if(incr_count >= 640)
      {
      	incr_count = 0;
      }*/

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace coarsefreq */
} /* namespace gr */

