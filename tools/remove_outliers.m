function filtered_data = remove_outliers(data)
% Generate example data

skip_condition = data(1, :) > 1;

valid_idx = ~skip_condition; 

filtered_data = data(:, valid_idx);  % 取有效的 z 值
end