function[A, b] = read_data()
    disp('starting to load dataset dynamic_share (~2 min)...');
    
    % download the dataset
    if not(isfolder('dynamic_share/'))
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00413/dataset.zip';
        filename = 'dynamic_share.zip';
        urlwrite(url, filename);
        unzip(filename, 'dynamic_share/');
    end
    
    input_file = dir('dynamic_share/*.txt');
    num_file = size(input_file);
    num_file = num_file(1);


    % pre-allocate the vectors
    size_A = 0;
    nnz_A = 0;
    for i = 1: num_file
        fname = strcat('dynamic_share/', input_file(i).name);
        fid = fopen(fname, 'r');
        filetext = fscanf(fid, '%c');
        nnz_A = nnz_A + size(regexp(filetext, ':', 'match'), 2);
        size_A = size_A + size(splitlines(filetext), 1) - 1;
        fclose(fid);
    end
    b = zeros(size_A, 1);
    col_idx = zeros(nnz_A, 1);
    row_idx = zeros(nnz_A, 1);
    val = zeros(nnz_A, 1);
    row = 1;
    b_ptr = 1;
    val_ptr = 1;
    col_ptr = 1;
    row_ptr = 1;

    % read data into A and b
    tStart = tic;
    for i = 1: num_file
        fname = strcat('dynamic_share/', input_file(i).name);
        fid = fopen(fname, 'r');

        % read the first number in each line to form the vector b
        tem_b = cell2mat( textscan(fid, '%f%*[^\n]') );
        b(b_ptr : b_ptr + size(tem_b, 1) - 1, :) = tem_b;
        b_ptr = b_ptr + size(tem_b, 1);

        % read indexes into col_idx
        frewind(fid);
        filetext = fscanf(fid, '%c');
        tem_col_idx = transpose(str2double(regexp(filetext, '[0-9]*(?=:)', 'match')));
        col_idx(col_ptr : col_ptr + size(tem_col_idx,1) - 1, :) = tem_col_idx;
        col_ptr = col_ptr + size(tem_col_idx,1);

        % read values into val
        tem_val = transpose(str2double(regexp(filetext, '(?<=:)[0-9]*', 'match')));
        val(val_ptr : val_ptr + size(tem_val,1) - 1, :) = tem_val;
        val_ptr = val_ptr + size(tem_val,1);

        % read the number of entries per row, construct row_idx
        filelines = splitlines(filetext);
        for t = 1 : (size(filelines, 1) - 1)
            line = split(filelines(t), ' ');
            num = size(line, 1) - 2;
            row_idx(row_ptr : row_ptr + num - 1, :) = row;
            row = row + 1;
            row_ptr = row_ptr + num;
        end

        fclose(fid);
    end
    
    tEnd = toc(tStart);
    disp("load data time:");
    disp(tEnd);

    col_idx = col_idx + 1;
    A = sparse(row_idx, col_idx, val, size_A, 482);
    A = full(A);
end