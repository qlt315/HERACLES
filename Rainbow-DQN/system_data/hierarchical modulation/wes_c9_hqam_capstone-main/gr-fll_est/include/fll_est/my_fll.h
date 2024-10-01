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

#ifndef INCLUDED_FLL_EST_MY_FLL_H
#define INCLUDED_FLL_EST_MY_FLL_H

#include <fll_est/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace fll_est {
  
  typedef enum {
    THRESHOLD_DYNAMIC,
    THRESHOLD_ABSOLUTE,
  } tm_type;


    /*!
     * \brief <+description of block+>
     * \ingroup fll_est
     *
     */
    class FLL_EST_API my_fll : virtual public gr::sync_block
    {
     public:
      typedef boost::shared_ptr<my_fll> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of fll_est::my_fll.
       *
       * To avoid accidental use of raw pointers, fll_est::my_fll's
       * constructor is in a private implementation
       * class. fll_est::my_fll::make is the public interface for
       * creating new instances.
       */
      static sptr make(const std::vector<gr_complex>& symbols,
                     float sps,
                     unsigned int mark_delay,
                     float threshold = 0.9,
                     tm_type threshold_method = THRESHOLD_ABSOLUTE);

    virtual std::vector<gr_complex> symbols() const = 0;
    virtual void set_symbols(const std::vector<gr_complex>& symbols) = 0;

    virtual unsigned int mark_delay() const = 0;
    virtual void set_mark_delay(unsigned int mark_delay) = 0;

    virtual float threshold() const = 0;
    virtual void set_threshold(float threshold) = 0;
    };

  } // namespace fll_est
} // namespace gr

#endif /* INCLUDED_FLL_EST_MY_FLL_H */

