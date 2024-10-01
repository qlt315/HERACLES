function [ QPSKphase ] = QPSK_BitToPhase( bit1,bit2 ) 
% This function maps the input bits to their respective phases using QPSK 
if bit1==0 && bit2==0
    QPSKphase=pi/4;
elseif bit1==0 && bit2==1
    QPSKphase=3*(pi/4);
elseif bit1==1 && bit2==1
    QPSKphase=5*(pi/4);    
elseif bit1==1 && bit2==0
    QPSKphase=7*(pi/4);   

end

