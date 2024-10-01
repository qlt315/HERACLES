% Multi-layer hierarchical modulation (currently 4 layers)

clear;
max_runs = 6000;  % Equals to block numbers
max_decode_iterations = 20;

min_sum = 1;
n_0 = 1/2;

max_power = 1; % Maximum transmit power
layer_num = 4; % Number of HM layers

Est_err_para = 0.5;

power = [0.4,0.3,0.2,0.1]; % Transmit power list
bit_num = [1296,1296,1296,1296];  % block length -> Should be one of 648, 1296, and 1944
rate = 1/2; % LDPC coding rate -> Should be one of 1/2, 2/3, 3/4, 5/6
info_length = bit_num * rate;

%Should be one of 'bpsk', 'ask4', 'ask8' (equivalently QPSK, 16-QAM, and 64-QAM)
constellation_name = ["bpsk","bpsk","bpsk","bpsk"];
modulation = cell(1,layer_num);
ldpc_code = cell(1,layer_num);
for i=1:layer_num
    modulation{i} = tm_constellation(constellation_name(i));
    ldpc_code{i} = ldpc(bit_num(i), bit_num(i)*rate);
end


ebno_db_min = 0;
ebno_db_inter = 0.5;
ebno_db_max = 10;
ebno_db_vec = (ebno_db_min:ebno_db_inter:ebno_db_max);


num_block_err = zeros(length(ebno_db_vec), layer_num);  % PER of each layer
num_block_err_hm = zeros(length(ebno_db_vec), 1); % Average PER of HM

num_bit_err = zeros(length(ebno_db_vec), layer_num);  % BER of each layer
num_bit_err_hm = zeros(length(ebno_db_vec), 1); % Average BER of HM

tic

disp(['Running ', num2str(layer_num), ' layer Hierarchical Modulation with LDPC coding rate = ', num2str(rate)]);
disp("Channel Estimation error = "); disp(num2str(Est_err_para));
disp("Power ratio = "); disp(num2str(power));
disp('Bit number = '); disp(num2str(bit_num));
disp('modulation method = '); disp(constellation_name);

snr_db_vec = ebno_db_vec + 10*log10(rate);

for i=1:layer_num
    ldpc_code{i}.load_wifi_ldpc(bit_num(i), rate);
    snr_db_vec = snr_db_vec + 10*log10(modulation{i}.n_bits);
end

for i_run = 1 : max_runs


    disp(['Current run = ', num2str(i_run), ' percentage complete = ', num2str((i_run-1)/max_runs * 100), '%', ' time elapsed = ', num2str(toc), ' seconds']);

    
    noise = sqrt(n_0) * randn(bit_num(i)/modulation{i}.n_bits, 1);
    
    info_bits = cell(1,layer_num);
    coded_bits = cell(1,layer_num);
    scrambling_bits = cell(1,layer_num);
    scrambled_bits = cell(1,layer_num);
    x = cell(1,layer_num);
    x_hm = zeros(size(noise));

    for i=1:layer_num
        info_bits{i} =  rand(info_length(i), 1) < 0.5;
        coded_bits{i} = ldpc_code{i}.encode_bits(info_bits{i});
        scrambling_bits{i} = (rand(bit_num(i), 1) < 0.5);
        scrambled_bits{i} = mod(coded_bits{i} + scrambling_bits{i}, 2);
        x{i} = modulation{i}.modulate(scrambled_bits{i});
        x_hm = x_hm + power(i) * x{i};
    end
    
    for i_snr = 1 : length(snr_db_vec)
        
        snr_db = snr_db_vec(i_snr);
        
        snr = 10^(snr_db/10);
        
        % h = randn(size(x)) + 1i * randn(size(x)) / sqrt(2);
        h = raylrnd(1 / sqrt(2), 1) * sqrt(snr_db); 
        
        y =  h .* x_hm + noise/sqrt(snr);

        % Channel Equalization
        Est_err = Est_err_para * abs(h);
        hEst = h + Est_err + Est_err * 1i;
        y = y ./ hEst;

        % SIC Demodulation
        llr = cell(1,layer_num);
        decoded_codeword = cell(1,layer_num);
        re_scrambled_bits = cell(1,layer_num-1);
        re_modu_bits = cell(1,layer_num-1);
        
        % Demodulate other layers with SIC
        y_sic = y;
        for i=1:layer_num

            [llr{i}, ~] = modulation{i}.compute_llr(y_sic, n_0/snr);

            llr{i} = llr{i} .* ( 1 - 2 * scrambling_bits{i});

            % decoded codeword (for WiFi codes, first K bits are the information bits)
            [decoded_codeword{i}, ~] = ldpc_code{i}.decode_llr(llr{i}, max_decode_iterations, min_sum);
            if i<layer_num
                re_scrambled_bits{i} = mod(decoded_codeword{i} + scrambling_bits{i}, 2);
                re_modu_bits{i} = modulation{i}.modulate(re_scrambled_bits{i});
                y_sic = y_sic - power(i) * re_modu_bits{i};
            end
        end

        % Calculate the PER 
        for i=1:layer_num
            if any(decoded_codeword{i} ~= coded_bits{i})
                num_block_err(i_snr,i) = num_block_err(i_snr,i) + 1;
                num_block_err_hm(i_snr) = num_block_err_hm(i_snr) + 1;
            else
                break; % Assumes the codeword will be decoded correctly for a higher SNR as well
            end
        end

        % Calculate the BER
        for i=1:layer_num
            for j=1:bit_num(i)
                if decoded_codeword{i}(j) ~= coded_bits{i}(j)
                    num_bit_err(i_snr,i) = num_bit_err(i_snr,i) + 1;
                    num_bit_err_hm(i_snr) = num_bit_err_hm(i_snr) + 1;
                end
            end
        end



    end
end

bler_hm = num_block_err_hm / max_runs;
bler = num_block_err / max_runs;

ber_hm = num_bit_err_hm / (max_runs * sum(bit_num));
ber = zeros(length(ebno_db_vec), layer_num);
for i=1:length(ebno_db_vec)
    for j=1:layer_num
        ber(i,j) = num_bit_err(i,j) / (max_runs * bit_num(j));
    end
end


file_name = strcat("four_layers_data/snr_", num2str(ebno_db_min), "_", num2str(ebno_db_inter), "_", num2str(ebno_db_max), "_", num2str(constellation_name(1)), "_", ...
    num2str(constellation_name(2)), "_", num2str(constellation_name(3)), "_", num2str(constellation_name(4)), "_esterr_", num2str(Est_err_para), "_rate_",num2str(rate), "_power_ratio_", ...
    num2str(power(1)), "_", num2str(power(2)), "_", num2str(power(3)), "_", num2str(power(4)), ".mat");
save(file_name);

mark_set = ["-o","-d","-*","-x"];

legend_set = strings(layer_num+1, 1);
for i=1:layer_num
    str1 = strcat("Layer ",num2str(i));
    legend_set(i) = strcat(str1,constellation_name(i));
end
legend_set(layer_num+1) = "HM average";




figure(1)
for i=1:layer_num
    semilogy(ebno_db_vec, bler(:,i), mark_set(i), 'LineWidth', 2, 'Markersize', 7); hold on;
end
semilogy(ebno_db_vec, bler_hm, "-^", 'LineWidth', 2, 'Markersize', 7,'Color',[0.13 0.55 0.13]); hold on;
grid on;
xlabel('SNR (dB)')
ylabel('Packet Error Rate')
legend(legend_set);
set(gca,'FontName','Times New Roman','FontSize',12);

figure(2)
for i=1:layer_num
    semilogy(ebno_db_vec, ber(:,i), mark_set(i), 'LineWidth', 2, 'Markersize', 7); hold on;
end
semilogy(ebno_db_vec, ber_hm, "-^", 'LineWidth', 2, 'Markersize', 7,'Color',[0.13 0.55 0.13]); hold on;
grid on;
xlabel('SNR (dB)')
ylabel('Bit Error Rate')
legend(legend_set);
set(gca,'FontName','Times New Roman','FontSize',12);

