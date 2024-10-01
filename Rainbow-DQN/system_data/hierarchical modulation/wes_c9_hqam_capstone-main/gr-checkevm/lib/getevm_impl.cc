/* -*- c++ -*- */
/*
 * Copyright 2022 Ashwini Bhagat.
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
#include "getevm_impl.h"
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
  namespace checkevm {

    getevm::sptr
    getevm::make(float max_evm)
    {
      return gnuradio::get_initial_sptr
        (new getevm_impl(max_evm));
    }


    /*
     * The private constructor
     */
    getevm_impl::getevm_impl(float max_evm)
      : gr::sync_block("getevm",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(2, 2, sizeof(unsigned char))),
              d_max_evm(max_evm),
              err_vec(0),
              hp(0),
              lp(0)
              
    {
    char hp,lp;
    }

    /*
     * Our virtual destructor.
     */
    getevm_impl::~getevm_impl()
    {
    }
    
    float getevm_impl::check_evm(const gr_complex &s)
    {
    	const float pi = 3.1416;
    	const float a = 1/sqrt(10);
    	const float b = 3/sqrt(10);
    	const float d_bound = 2/sqrt(10.);
    	gr_complex s_est;
    	float err_vec;
    	
    	
    	
    	
    	//estimate the symbol based on input value 
    	if ((s.real()>=0)&&(s.imag()>=0)) //Top right quadrant 
    	{
    		hp = 0;
    		if ((s.real()>=d_bound)&&(s.imag()>=d_bound))  // Top right subquadrant
    		{
    			s_est.real(b);s_est.imag(b);lp = 0;
    		} else if ((s.real()<d_bound)&&(s.imag()>=d_bound))  // top left subquadrant
    		{
    			s_est.real(a);s_est.imag(b);lp = 1;
    		} else if ((s.real()<d_bound)&&(s.imag()<d_bound))   // bottom left subquadrant 
    		{
    			s_est.real(a);s_est.imag(a);lp = 3;
    		} else if ((s.real()>=d_bound)&&(s.imag()<d_bound))  // bottom right subquadrant
    		{
    			s_est.real(b);s_est.imag(a);lp = 2;
    		}
    		
    	} else if ((s.real()<0)&&(s.imag()>=0)) //top left quadrant 
    	{
    		hp = 1;
    		if ((s.real() >= -d_bound)&&(s.imag()>=d_bound))  // Top right subquadrant
    		{
    			s_est.real(-a);s_est.imag(b);lp = 0; 
    		} else if ((s.real() < -d_bound)&&(s.imag()>=d_bound))  // Top left subquadrant
    		{
    			s_est.real(-b);s_est.imag(b);lp = 1;
    		} else if ((s.real()< -d_bound)&&(s.imag()<d_bound))   // Bottom left subquadrant 
    		{
    			s_est.real(-b);s_est.imag(a);lp = 3;
    		} else if ((s.real()>= -d_bound)&&(s.imag()<d_bound))  // Bottom right subquadrant
    		{
    			s_est.real(-a);s_est.imag(a);lp = 2;
    		}
    	} else if ((s.real()<0)&&(s.imag()<0))  //bottom left quadrant 
    	{
    		hp = 3;
    		if ((s.real() >= -d_bound)&&(s.imag()>= -d_bound))  // top right subquadrant
    		{
    			s_est.real(-a);s_est.imag(-a);lp = 0;
    		} else if ((s.real() < -d_bound)&&(s.imag()>= -d_bound))  // top left subquadrant
    		{
    			s_est.real(-b);s_est.imag(-a);lp = 1;
    		} else if ((s.real()< -d_bound)&&(s.imag()< -d_bound))   // bottom left subquadrant 
    		{
    			s_est.real(-b);s_est.imag(-b);lp = 3;
    		} else if ((s.real()>= -d_bound)&&(s.imag()< -d_bound))  // bottom right subquadrant
    		{
    			s_est.real(-a);s_est.imag(-b);lp = 2;
    		}
    	} else if ((s.real()>=0)&&(s.imag()<0))  //bottom right quadrant 
    	{
    		hp = 2;
    		if ((s.real() >= d_bound)&&(s.imag()>= -d_bound))  // top right subquadrant
    		{
    			s_est.real(b);s_est.imag(-a);lp = 0;
    		} else if ((s.real() < d_bound)&&(s.imag()>= -d_bound))  // top left subquadrant
    		{
    			s_est.real(a);s_est.imag(-a);lp = 1;
    		} else if ((s.real()< d_bound)&&(s.imag()< -d_bound))   // bottom left subquadrant 
    		{
    			s_est.real(a);s_est.imag(-b);lp = 3;
    		} else if ((s.real()>= d_bound)&&(s.imag()< -d_bound))  // bottom right subquadrant
    		{
    			s_est.real(b);s_est.imag(-b);lp = 2;
    		}
    	}
    	
    	err_vec = sqrt(pow((s_est.imag()-s.imag()),2)+pow((s_est.real()-s.real()),2));
    	

    	
 	return err_vec;
    }
    

    int
    getevm_impl::work(int noutput_items,
        gr_vector_const_void_star &input_items,
        gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      unsigned char *out_hp = (unsigned char *) output_items[0];
      unsigned char *out_lp = (unsigned char *) output_items[1];
      float m_evm = 0.0;
      const float pi = 3.1416;
      const float a = 1/sqrt(10);
      const float b = 3/sqrt(10);
      const float d_bound = 2/sqrt(10.);
      
      

      // Do <+signal processing+>
      for(int i = 0; i < noutput_items ; i++)
      {
      	
      	m_evm = check_evm(in[i]);
      	
      	if(m_evm < d_max_evm)
      	{
      		//printf("I am here greater than threshold\n");
      		out_hp[i] = hp;
      		out_lp[i] = lp;
      	}
      	else 
      	{
      		//printf("I am here less than threshold\n");
      		out_hp[i] = hp;
      		out_lp[i] = 0;
      	}
      	//printf("EVM is %f\n", m_evm);
    	//printf("hp is %c and lp is %c\n", out_hp[i],out_lp[i]);
      	
      }
      

      // Tell runtime system how many output items we produced.
      return noutput_items;
    }

  } /* namespace checkevm */
} /* namespace gr */

