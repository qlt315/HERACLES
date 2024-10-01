# wes_c9_hqam_capstone
# Hierarchical Modulation - Capstone Project - UCSD WES Cohort 9 

In a typical broadcast system, there is a single transmitter and multiple receivers and some receivers are in high SNR condition and some low SNR condition. Usually there is no feedback for channel condition for each Rx, usually the Tx has to modulate data for the worst channel condition reception.(Say QPSK) This seems okay but a lot of Rx out there are in good channel conditions and perhaps have capility to decode higher modulation schemes. 

This is where Hierarchical Modulation comes into picture. The concept is to divide your transmitter data streams into multiple priority stream and modulate each stream with either different or same modulation scheme. 

For our project we modulated our HP(high priority)/Base Layer data with QPSK (Gray Coded) and we modulated our LP(Low Priority)/Enhancement Layer data with QPSK around each HP constellation points. Please see below diagram.   

<img width="325" alt="image" src="https://user-images.githubusercontent.com/92651382/172268700-7ef25cd0-016e-4404-a8cf-20ebea967e06.png">


Please see below block diagram for HQAM:-

<img width="750" alt="image" src="https://user-images.githubusercontent.com/92651382/170805401-b8c40dd2-c099-4f38-97b6-4311f0307585.png">

# MATLAB Simulation 
We take our input image and perform DCT on it to give us high priority stream and low priority stream. These streams are now passed thorugh channel encoder to create redundancies. We are using Convolution Encoder and Decoder of rate 1/3. Each channel encoded stream is separately QPSK modulated and scaled to create HQAM constellation. (Shown above)
For simulation purpose, we have added AWGN and create a for loop to sweep through SNR from 1-20 dB. 

Below are received constellation at an SNR of 10 dB and 20 dB. 

<img width="393" alt="image" src="https://user-images.githubusercontent.com/92651382/170805794-4570e176-9fc8-4e39-956a-cdd42b98ab36.png">
<img width="392" alt="image" src="https://user-images.githubusercontent.com/92651382/170805772-bd6c5f16-bf1a-42d1-8012-45cb9a896785.png">

Below are the final images at the reeceiver side:-
<img width="747" alt="image" src="https://user-images.githubusercontent.com/92651382/170805953-c6e5a4d3-bb9b-40d0-a0e2-8b7d885498e2.png">

The MATLAB code is under MATLAB folder. It has a sample "Lenna" image to process through the simulation. (Size of image is 128x128).

# GNU Radio - HQAM Implementation 

----- Please note that you need to install UHD first, then volk, then GNU Radio and then RFNoc -------------

We have uploaded a pdf document(GNURadio Dependencies) on repo to help with installation of dependencies and all. We are using standard GNU Radio version 3.8.5 which can be downloaded from gnuradio.org and refer to below link for installation instructions. 

https://wiki.gnuradio.org/index.php/InstallingGR

The folder structure of HQAM Project is following:-
1. wes_c9_hqam_capstone/MATLAB
2. wes_c9_hqam_capstone/gnu_radio_flowgraphs
3. wes_c9_hqam_capstone/gr-checkevm
4. wes_c9_hqam_capstone/gr-dd_pll
5. wes_c9_hqam_capstone/gr-fll_est
6. wes_c9_hqam_capstone/gr-dct
7. wes_c9_hqam_capstone/gr-hqam
8. wes_c9_hqam_capstone/gr-coarsefreq
9. wes_c9_hqam_capstone/rfnoc-capstone

gnu_radio_flowgraphs have all the GRC files which has HQAM Tx and Rx design. 
There are a lot of GRCs which we created and used during on testing to validate each functionality. We tried to name GRCs for testing purpose as *_test_*.grc to keep it clear. To run the project please follow the instructions after installing everything above. 

1. download all the OOT modules and build them. 
  - To build create a build directory inside the gr-* folder using "mkdir build".
  - cd build
  - cmake ../
  - make 
  - sudo make install 
  - sudo ldconfig 
2. now copy all the GRCs in a folder(any folder)
3. Go to Terminal and type gnuradio-companion 
4. click open tab and import GRCs. 
5. You need to run all the hierarchical blocks and reload the main GRCs before you can run the test. 








