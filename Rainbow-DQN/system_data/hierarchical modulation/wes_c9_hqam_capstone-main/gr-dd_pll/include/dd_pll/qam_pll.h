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

#ifndef INCLUDED_DD_PLL_QAM_PLL_H
#define INCLUDED_DD_PLL_QAM_PLL_H

#include <dd_pll/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace dd_pll {

    /*!
     * \brief <+description of block+>
     * \ingroup dd_pll
     *
     */
    class DD_PLL_API qam_pll : virtual public gr::block
    {
    
     private:
     	
     	
     public:
      typedef boost::shared_ptr<qam_pll> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of dd_pll::qam_pll.
       *
       * To avoid accidental use of raw pointers, dd_pll::qam_pll's
       * constructor is in a private implementation
       * class. dd_pll::qam_pll::make is the public interface for
       * creating new instances.
       */
      static sptr make(float zeta, float fn, double sample_freq);
    };

  } // namespace dd_pll
} // namespace gr

#endif /* INCLUDED_DD_PLL_QAM_PLL_H */

