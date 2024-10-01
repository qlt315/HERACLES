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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "qam_pll_impl.h"
#include <gnuradio/expj.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/math.h>
#include <gnuradio/sincos.h>
#include <boost/format.hpp>
#include <gnuradio/blocks/complex_to_arg.h>
#include <gnuradio/blocks/conjugate_cc.h>
#include <cmath>
#include <iostream>

using namespace std;

namespace gr {
  namespace dd_pll {

    qam_pll::sptr
    qam_pll::make(float zeta, float fn, double sample_freq)
    {
      return gnuradio::get_initial_sptr
        (new qam_pll_impl(zeta, fn, sample_freq));
    }


    /*
     * The private constructor
     */
    qam_pll_impl::qam_pll_impl(float zeta, float fn, double sample_freq)
      : gr::block("qam_pll",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
              d_zeta(zeta),
              d_fn(fn),
              d_sample_freq(sample_freq),
              f_in(0),
              f_int_out(0),
              f_in_last(0),
              f_out_last(0),
              f_out(0),
              f_int_out_last(0),
              vco_in(0),
              vco_in_last(0),
              vco_out(0),
              vco_out_last(0)
    {}

    /*
     * Our virtual destructor.
     */
    qam_pll_impl::~qam_pll_impl()
    {
    }

    void
    qam_pll_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      
      unsigned ninputs = ninput_items_required.size ();
      
      //cout << "noutput_items forecast -- " <<noutput_items << endl;
      for(unsigned i = 0; i < ninputs; i++)
      	ninput_items_required[i] = noutput_items; 
    }
    
    gr_complex qam_pll_impl::call_pll(const gr_complex &sample)
    {
    	float s2, s3;
    	const float pi = 3.1416;
    	const float a = 1/sqrt(10);
    	const float b = 3/sqrt(10);
    	const float d_bound = 2./sqrt(10.);
    	float T = 1/d_sample_freq;
    	gr_complex s_est;
    	gr_complex s;
    	
    
    	// loop gain Kt 
    	float Kt = 4*pi*d_zeta*d_fn;
    	// loop filter parameter/constant Ka
    	float Ka = pi*d_fn/d_zeta;
    	
    	//cout << "Kt is " << Kt << endl;
    	//cout << "Ka is " << Ka << endl;
    	
    	// rotate the input sample 
    	s = sample*gr_expj(-vco_out);
    	//cout << "sample is " << sample << endl;
    	//cout << "vco out is " << vco_out << endl;
    	//cout << "gr_expj is " << gr_expj(-vco_out) << endl;
    	//cout << "s is " << s << endl;
    	
    	
    	
    	//check angle of sample 
    	//float s1 = std::atan2(sample.imag(),sample.real());
    	//cout << s1 << endl;
    	
    	//estimate the symbol based on input value 
    	if ((s.real()>=0)&&(s.imag()>=0)) //first quadrant 
    	{
    		if ((s.real()>=d_bound)&&(s.imag()>=d_bound))  // first subquadrant
    		{
    			s_est.real(b);s_est.imag(b);
    		} else if ((s.real()<d_bound)&&(s.imag()>=d_bound))  // second subquadrant
    		{
    			s_est.real(a);s_est.imag(b);
    		} else if ((s.real()<d_bound)&&(s.imag()<d_bound))   // Third subquadrant 
    		{
    			s_est.real(a);s_est.imag(a);
    		} else if ((s.real()>=d_bound)&&(s.imag()<d_bound))  // fourth subquadrant
    		{
    			s_est.real(b);s_est.imag(a);
    		}
    	} else if ((s.real()<0)&&(s.imag()>=0)) //second quadrant 
    	{
    		if ((s.real() >= -d_bound)&&(s.imag()>=d_bound))  // first subquadrant
    		{
    			s_est.real(-a);s_est.imag(b);
    		} else if ((s.real() < -d_bound)&&(s.imag()>=d_bound))  // second subquadrant
    		{
    			s_est.real(-b);s_est.imag(b);
    		} else if ((s.real()< -d_bound)&&(s.imag()<d_bound))   // Third subquadrant 
    		{
    			s_est.real(-b);s_est.imag(a);
    		} else if ((s.real()>= -d_bound)&&(s.imag()<d_bound))  // fourth subquadrant
    		{
    			s_est.real(-a);s_est.imag(a);
    		}
    	} else if ((s.real()<0)&&(s.imag()<0))  //third quadrant 
    	{
    		if ((s.real() >= -d_bound)&&(s.imag()>= -d_bound))  // first subquadrant
    		{
    			s_est.real(-a);s_est.imag(-a);
    		} else if ((s.real() < -d_bound)&&(s.imag()>= -d_bound))  // second subquadrant
    		{
    			s_est.real(-b);s_est.imag(-a);
    		} else if ((s.real()< -d_bound)&&(s.imag()< -d_bound))   // Third subquadrant 
    		{
    			s_est.real(-b);s_est.imag(-b);
    		} else if ((s.real()>= -d_bound)&&(s.imag()< -d_bound))  // fourth subquadrant
    		{
    			s_est.real(-a);s_est.imag(-b);
    		}
    	} else if ((s.real()>=0)&&(s.imag()<0))  //fourth quadrant 
    	{
    		if ((s.real() >= d_bound)&&(s.imag()>= -d_bound))  // first subquadrant
    		{
    			s_est.real(b);s_est.imag(-a);
    		} else if ((s.real() < d_bound)&&(s.imag()>= -d_bound))  // second subquadrant
    		{
    			s_est.real(a);s_est.imag(-a);
    		} else if ((s.real()< d_bound)&&(s.imag()< -d_bound))   // Third subquadrant 
    		{
    			s_est.real(a);s_est.imag(-b);
    		} else if ((s.real()>= d_bound)&&(s.imag()< -d_bound))  // fourth subquadrant
    		{
    			s_est.real(b);s_est.imag(-b);
    		}
    	}
    	
    	// do conjugate of estimated symbol 
    	//float temp_i = s_est.imag();
    	//s_est.imag(-temp_i);
    	
    	//cout << "the s_est is " << s_est << endl;
    	s_est = std::conj(s_est);
    	//cout << "the s_est conjugate is " << s_est << endl;
    	
    	
    	gr_complex temp_c = s*s_est;
    	//s2 = std::atan2(temp_c.imag(),temp_c.real());
    	s2 = std::arg(temp_c);
    	
    	
    	// apply loop gain 
    	s3 = Kt*s2;
    	//cout << "the s3 is " << s3 << endl;
    	
    	f_in = s3; // feed input to loop
    	
    	//cout << "the f_in is " << f_in << endl;
    	
    	f_int_out = f_int_out_last + Ka*(T/2)*(f_in + f_in_last); 
    	f_out = f_in + f_int_out;
    	
    	//cout << "the f_int_out is " << f_int_out << endl;
    	//cout << "the f_out (vco_in) is " << f_out << endl;
    	//cout << "the f_int_out_last is " << f_int_out_last << endl;
    	//cout << "the f_in_last is " << f_in_last << endl;
    	
    	
    	// Output of loop filter is last value pulse a discrete form of the 
    	// integration of the input
    	
    	f_in_last = f_in;     
    	f_out_last = f_out;   
    	vco_in = f_out;       
    	f_int_out_last = f_int_out;
    	
    	// VCO input is sum of scaled output from phase detector 
    	// and the integrating loop filter
    
    	vco_out = vco_out_last + (T/2)*(vco_in + vco_in_last);
    
    	// VCO integrates the phase - same form as integrator for 
    	// the loop filter
    	vco_in_last = vco_in;			//Update
    	vco_out_last = vco_out;		//Update
    	//cout << "second vco " << vco_out << endl;
    	//cout << "T " << T << endl;
    	
    	
    	return s;
    
    }

    int
    qam_pll_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];
      
      //cout << "noutput_items general work -- " <<noutput_items << endl;
      
      // Do <+signal processing+>
      for(int i = 0; i < noutput_items ; i++)
      {
      	
      	out[i] = call_pll(in[i]);
      
      }
      
      
      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each (noutput_items);

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }
    

  } /* namespace dd_pll */
} /* namespace gr */

