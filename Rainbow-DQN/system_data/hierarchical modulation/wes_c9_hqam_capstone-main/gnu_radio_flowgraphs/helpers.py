# this module will be imported in the into your flowgraph


def unpack_values(values_in, bits_per_value, bits_per_symbol):   
    # verify that 8 is divisible by bits_per_symbol 
    m = bits_per_value / bits_per_symbol
    # print(m)
    mask = 2**(bits_per_symbol)-1
        
    if bits_per_value != m*bits_per_symbol:
        print("error - bits per symbols must fit nicely into bits_per_value bit values")
        return []
        
    num_values = len(values_in)
    # print(num_values)
    num_symbols = int(num_values*( m) )
    # print(num_symbols)
    cur_byte = 0
    cur_bit = 0
    out = []
    for i in range(num_symbols):
        s = (values_in[cur_byte] >> (bits_per_value-bits_per_symbol-cur_bit)) & mask
        out.append(s)
        cur_bit += bits_per_symbol
        
        if cur_bit >= bits_per_value:
            cur_bit = 0
            cur_byte += 1
            
        # if cur_byte >= num_values:
            # break;
            
    return out
    
def map_symbols_to_constellation(symbols, cons):
    # print("symbols:")
    # print(symbols)
    # print("points:")
    # print(cons.points())
    l = list(map(lambda x: cons.points()[x], symbols))
    # print("Training Symbols")
    # print(l)
    return l
