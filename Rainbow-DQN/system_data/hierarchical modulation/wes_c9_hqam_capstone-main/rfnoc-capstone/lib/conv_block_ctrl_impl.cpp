/* -*- c++ -*- */
/* 
 * Copyright 2022 <+YOU OR YOUR COMPANY+>.
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


#include <capstone/conv_block_ctrl.hpp>
#include <uhd/rfnoc/registry.hpp>

using namespace uhd::rfnoc;

// Note: Register addresses should increment by 4
const uint32_t conv_block_ctrl::REG_USER_ADDR    = 0;
const uint32_t conv_block_ctrl::REG_USER_DEFAULT = 0;

class conv_block_ctrl_impl : public conv_block_ctrl
{
public:
    RFNOC_BLOCK_CONSTRUCTOR(conv_block_ctrl)
    {
        _register_props();
    }
private:
    void _register_props()
    {
        register_property(&_user_reg, [this]() {
            int user_reg = this->_user_reg.get();
            this->regs().poke32(REG_USER_ADDR, user_reg);
        });

        // register edge properties
        register_property(&_type_in);
        register_property(&_type_out);

        // add resolvers for type (keeps it constant)
        add_property_resolver({&_type_in}, {&_type_in}, [& type_in = _type_in]() {
            type_in.set(IO_TYPE_SC16);
        });
        add_property_resolver({&_type_out}, {&_type_out}, [& type_out = _type_out]() {
            type_out.set(IO_TYPE_SC16);
        });
    }

    property_t<int> _user_reg{"user_reg", REG_USER_DEFAULT, {res_source_info::USER}};

    property_t<std::string> _type_in = property_t<std::string>{
        PROP_KEY_TYPE, IO_TYPE_SC16, {res_source_info::INPUT_EDGE}};
    property_t<std::string> _type_out = property_t<std::string>{
        PROP_KEY_TYPE, IO_TYPE_SC16, {res_source_info::OUTPUT_EDGE}};

};

UHD_RFNOC_BLOCK_REGISTER_DIRECT(conv_block_ctrl, 0x8888, "conv", CLOCK_KEY_GRAPH, "bus_clk");
