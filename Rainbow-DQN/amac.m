% Baseline -- Adaptive Modulation and Coding (AMAC)
clear; clc;
rng(666);
% Parameter settings
slot_num = 3000; % Number of time slots
max_energy = 2000; % Maximum energy consumption (J)
curr_energy = max_energy; % Available energy of current slot
last_energy = 0; % Consumed energy of last slot
sensor_num = 4; % Number of sensors
bandwidth = 20e6; % System bandwidth (Hz)

max_power = 1; % Maximum transmit power (W)
Est_err_para = 0.5;

h = (randn(slot_num, 1) + 1i * randn(slot_num, 1)) / sqrt(2);
% Channel estimation
Est_err = Est_err_para * abs(h);
hEst = h + Est_err + Est_err * 1i;
noise = sqrt(1/2) * randn(1,1);

data_size = zeros(1,sensor_num);
% Parameter settings of each slot
    for j=1:sensor_num
        data_size(j) = rand(1) * 5000;  % Quantized data size (bits)
    end

total_delay_list = zeros(1,slot_num);
total_energy_list = zeros(1,slot_num);
total_acc_list = zeros(1,slot_num);
reward_list = zeros(1,slot_num);

delay_vio_num = 0;
re_trans_num = NaN;
consider_re_trans = 0;  % 0 or 1

context_list = ["snow", "fog", "motorway", "night","rain","sunny"];
curr_context = context_list(randi(length(context_list)));
context_interval = 1000; % Interval for context to change 


% Data loading and fitting

load('platform_data.mat');

folder_path = "typical modulation";
file_list = dir(fullfile(folder_path, '*.mat'));
all_valid_file = {};



for i = 1:length(file_list)

    file_name = file_list(i).name;
    if contains(file_name, strcat("esterr_", num2str(Est_err_para)))
        all_valid_file = [all_valid_file, {file_name}];
    end
end

all_p = cell(1,length(all_valid_file));  % Polynomial coefficients
all_mod_method = cell(1,length(all_valid_file));
all_rate = cell(1,length(all_valid_file));

for i = 1:length(all_valid_file)
    file_name = all_valid_file{i};
    pattern = 'snr_([\d\._]+)_(\w+)_esterr_([\d\.]+)_rate_(\d+)_(\d+)';
    tokens = regexp(file_name, pattern, 'tokens');
    
    if ~isempty(tokens)
        snr_str = tokens{1}{1};
        snr_values = strsplit(snr_str, '_');
        snr_min = str2double(snr_values{1});   
        snr_int = str2double(snr_values{2});  
        snr_max = str2double(snr_values{3});   
        mod_method = tokens{1}{2};
        rate_num = str2double(tokens{1}{4});
        rate_den = str2double(tokens{1}{5});  
        rate = rate_num / rate_den;
    end
    
    directory = "typical modulation/";
    full_path = strcat(directory, file_name);
    ber_data = load(full_path);
    snr_list = snr_min:snr_int:snr_max;
    p = polyfit(snr_list, ber_data.bler, 5);  

    all_p{i} = p;
    all_mod_method{i} = mod_method;
    all_rate{i} = rate;

    
    % % Validate the fitting results
    % ber_fit = polyval(p, snr_list);
    % 
    % 
    % figure;
    % plot(snr_list, ber_data.bler, 'o', 'MarkerSize', 8, 'DisplayName', 'Raw Data');
    % hold on;
    % plot(snr_list, ber_fit, '-r', 'LineWidth', 2, 'DisplayName', 'Fitting Results');
    % xlabel('SNR (dB)');
    % ylabel('BER');
    % legend;
    % title('SNR-BER Pair Function');
    % grid on;
end




for i=1:slot_num


    if mod(i,context_interval) == 0
        curr_context = context_list(randi(length(context_list)));
    end

    curr_energy = curr_energy - last_energy;
    max_delay = 0.5;  % Maximum tolerant delay (s)


    % Calcualte SNR
    snr = abs(hEst(i)) * max_power / abs(noise);

    % Search every transmission scheme to find the optimal one
    acc_curr_list = zeros(1,length(all_valid_file));
    
    for j=1:length(all_valid_file)
        p_curr = all_p{j};
        mod_method_curr = all_mod_method{j};
        rate_curr = all_rate{j};
        
        



        % BER Calculation
        snr_dB = 10*log10(snr);
        ber_curr = polyval(p, snr_dB);
        if ber_curr > 1
            ber_curr = 1;
        elseif ber_curr < 0
            ber_curr = 0;
        end
        
        

        % Accuracy Calculation
        acc_curr = eval(strcat(curr_context,"(23,1)"));
        acc_curr_list(j) = acc_curr;

    end
    
    [acc_opt, opt_index] = max(acc_curr_list);
    
    rate_opt = all_rate{opt_index};

    % Delay Calculation
    trans_rate = bandwidth * log2(1+snr);

    coded_data_size = floor(sum(data_size)/rate_opt);

    trans_delay = coded_data_size / trans_rate;
    trans_energy = trans_delay * max_power;
    
    stem_com_delay = sum(stem_delay);
    stem_com_energy = sum(stem_energy);

    branch_com_delay = branch_delay(23,1);
    branch_com_energy = branch_energy(23,1);
    
    % PER Calcualtion
    per = 1-(1-ber_curr)^(coded_data_size);

    
    % Retransmission
    if consider_re_trans == 1
        re_trans_num = 0;
        trans_success = 0;
        while trans_success == 0
            if rand() < per
                re_trans_num = re_trans_num + 1;
                trans_success = 0;  % Transmission error
            else
                trans_success = 1;  % Transmission success
                disp("Transmission success !")
            end
        end
    end


    total_delay_list(1,i) = stem_com_energy + trans_delay + branch_com_delay;
    total_energy_list(1,i) = stem_com_energy + trans_energy + branch_com_energy;
    last_energy = total_energy_list(1,i);

    total_acc_list(1,i) = acc_opt;
    
    if total_delay_list(1,i) > max_delay
        delay_vio_num = delay_vio_num + 1;
    end
    
    reward_1 = acc_opt / 100;
    reward_2 = (max_delay - total_delay_list(1,i)) / max_delay;
    reward_3 = curr_energy / max_energy;
    reward = reward_1 + reward_2 + reward_3;
    reward_list(1,i) = reward;
    
end

    aver_total_delay = sum(total_delay_list) / slot_num;
    aver_total_energy = sum(total_energy_list) / slot_num;
    aver_acc = sum(total_acc_list) / slot_num;
    aver_reward = sum(reward_list) / slot_num;

    fprintf('Average Total Delay (s): %.2f\n', aver_total_delay);
    fprintf('Average Total Energy Consumption (J): %.2f\n', aver_total_energy);
    fprintf('Remaining Energy Consumption (J): %.2f\n', curr_energy);
    fprintf('Average Reward: %.2f\n', aver_reward);
    fprintf('Average Accuracy: %.2f\n', aver_acc);
    fprintf('Timeout Number: %.2f\n', delay_vio_num);
    fprintf('Retransmission Number: %.2f\n', re_trans_num);  % If not consider -> 'NaN'


    
    

    



