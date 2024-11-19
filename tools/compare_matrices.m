function [relative_diff, worse_diff] = compare_matrices(matrix_a, matrix_b)
    % Compare two matrices or cell arrays and compute the relative improvement/reduction
    % between each corresponding element, as well as the target value based on conditions.
    %
    % Parameters:
    %   matrix_a (MxN matrix or cell array): The first matrix/cell (algorithm A metrics).
    %   matrix_b (MxN matrix or cell array): The second matrix/cell (algorithm B metrics).
    %
    % Returns:
    %   relative_diff (MxN matrix or cell array): Matrix or cell of relative differences (%).
    %   target_value (scalar): Based on the conditions:
    %       - If all values are negative: Largest negative value.
    %       - If all values are positive: Smallest positive value.
    %       - If both positive and negative exist: Smallest value.
    
    % Check if inputs are cell arrays
    if iscell(matrix_a) && iscell(matrix_b)
        % Ensure the cell arrays have the same size
        assert(isequal(size(matrix_a), size(matrix_b)), ...
            'Cell arrays must have the same size.');
        
        % Initialize outputs
        relative_diff = cell(size(matrix_a));
        all_diff = []; % Store all differences for analysis
        
        % Iterate over each cell
        for i = 1:numel(matrix_a)
            % Convert cell elements to averages if they are arrays
            if isnumeric(matrix_a{i}) && numel(matrix_a{i}) > 1
                matrix_a{i} = mean(matrix_a{i}(:));
            end
            if isnumeric(matrix_b{i}) && numel(matrix_b{i}) > 1
                matrix_b{i} = mean(matrix_b{i}(:));
            end
            
            % Compute the relative difference for each cell
            relative_diff{i} = ((matrix_a{i} - matrix_b{i}) ./ abs(matrix_b{i})) * 100;
            
            % Collect the differences
            all_diff = [all_diff; relative_diff{i}(:)];
        end
    elseif isnumeric(matrix_a) && isnumeric(matrix_b)
        % Ensure matrices have the same size
        assert(isequal(size(matrix_a), size(matrix_b)), ...
            'Numeric matrices must have the same size.');
        
        % Calculate the relative difference
        relative_diff = ((matrix_a - matrix_b) ./ abs(matrix_b)) * 100;
        all_diff = relative_diff(:);
    else
        error('Both inputs must be either numeric matrices or cell arrays of numeric matrices.');
    end
    
    % Determine the target value
    if all(all_diff < 0)
        % All differences are negative, find the largest negative value
        worse_diff = max(all_diff);
    elseif all(all_diff > 0)
        % All differences are positive, find the smallest positive value
        worse_diff = min(all_diff);
    else
        % Mixed positive and negative, find the absolute smallest value
        [~, idx] = min(all_diff);
        worse_diff = all_diff(idx);
    end
end
