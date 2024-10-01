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

#ifndef INCLUDED_COARSEFREQ_C_FREQ_OFFSET_H
#define INCLUDED_COARSEFREQ_C_FREQ_OFFSET_H

#include <coarsefreq/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace coarsefreq {

    /*!
     * \brief <+description of block+>
     * \ingroup coarsefreq
     *
     */
    class COARSEFREQ_API c_freq_offset : virtual public gr::block
    {
     public:
      typedef boost::shared_ptr<c_freq_offset> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of coarsefreq::c_freq_offset.
       *
       * To avoid accidental use of raw pointers, coarsefreq::c_freq_offset's
       * constructor is in a private implementation
       * class. coarsefreq::c_freq_offset::make is the public interface for
       * creating new instances.
       */
      static sptr make(int packet_len, float threshold, double sample_freq);
    };

  } // namespace coarsefreq
} // namespace gr

#endif /* INCLUDED_COARSEFREQ_C_FREQ_OFFSET_H */

