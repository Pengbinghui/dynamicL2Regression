%-------------------------hyper-parameters------------------------------%
dataset = 'dynamic_share'; % dataset = {'dynamic_share', 'gaussian'}
method = 'apx_sampling'; % method = {'apx_sampling', 'woodbury', 'uniform', 'row_sampling'}
num_exp = 5; % number of repeated experiments

err = 0.1; % error parameter
k = round(1/(2* err^2)); % JL size, only used for apx_sampling
p = 0.2; % sampling probability, only used for uniform sampling
samplingScale = 1/(2*err^2); % scaled sampling probability, used to compute the sampling probability of row_sampling and apx_sampling


%---------------------------read/generate data-----------------------%
if isequal(dataset, 'gaussian')
    d = 500;
    T = 500000;
    
    A = normrnd(0, 1/sqrt(d), [d+T+1, d]);
    
    
    % Scale a random set of d/10 rows by sqrt(T).
    scale = ones(1, d+T+1);
    T_start = 50000;
    for i = 1 : d/10
        ind = randi([T_start + d + 1, T_start + d + 1 + T/10]);
        scale(1, ind) = sqrt(T);
    end
    
    A = transpose(scale).* A;
    
    x_opt = normrnd(0, 1/sqrt(d), [d,1]);
    b = A * x_opt + transpose(scale).* normrnd(0, 1/sqrt(d), [d+T+1,1]);
    
elseif isequal(dataset, 'dynamic_share')
    if exist('A_dynamic_share', 'var') == 0
        [A_dynamic_share, b_dynamic_share] = read_data();
    end
    A = A_dynamic_share;
    b = b_dynamic_share;
    d = size(A, 2);
    T = size(A, 1);
    sigma = 1;
    A = [eye(d) * sigma; zeros(1, d); A];
    b = [zeros(d, 1); sigma; b];
    
    T_start = 10000;

end


error_rate = 0; % error rate
sample_rate = 0; % sample rate
total_time = 0; % total runtime
error_std = norm(A * (inv(transpose(A) * A) * (transpose(A) * b)) - b)^2 / (T+d+1); % least-squares error


%---------------------------experiments--------------------------%
for exp_count = 1: num_exp
    
    % preprocess
    if isequal(method, 'apx_sampling')
        A_init = A(1 : d+1+T_start, :);
        b_init = b(1 : d+1+T_start);


        M = [A_init b_init];
        N = M;
        H = inv(transpose(N) * N);
        B = N * H;
        J = normrnd(0, 1/sqrt(k), [k, d+1+T_start]);
        JL_B = J * B;
        JL_N = J * N;

        G = inv(transpose(A_init) * A_init);
        u = transpose(A_init) * b_init;
        x = G * u;
    elseif isequal(method, 'woodbury')
        A_init = A(1 : d+1+T_start, :);
        b_init = b(1 : d+1+T_start);

        G = inv(transpose(A_init) * A_init);
        u = transpose(A_init) * b_init;
        x = G * u;

    elseif isequal(method, 'uniform')
        A_init = A(1 : d+1+T_start, :);
        b_init = b(1 : d+1+T_start);

        G = inv(transpose(A_init) * A_init);
        u = transpose(A_init) * b_init;
        x = G * u;

    elseif isequal(method, 'row_sampling')
        A_init = A(1 : d+1+T_start, :);
        b_init = b(1 : d+1+T_start);


        M = [A_init b_init];
        H = inv(transpose(M) * M);

        G = inv(transpose(A_init) * A_init);
        u = transpose(A_init) * b_init;
        x = G * u;
    end






    tStart = tic;
    count = 0;
    for t = T_start + 1 : T
        if isequal(method, 'apx_sampling')
            % new row of data
            a = transpose(A(d+1+t, :));
            label = b(d+1+t);
            m = [a; label];

            % leverage score sampling
            tau = norm(JL_B * m)^2;
            p = min(tau * samplingScale, 1); % sampling probability
            if rand > p
                continue;
            end


            count = count + 1;
            % update
            v1 = H * m /sqrt(p);
            v2 = 1 / (1 + transpose(m) * H * m / p);
            H = H - v1 * transpose(v1) * v2;


            new_JL_column = normrnd(0, 1/sqrt(k), [k, 1]);
            JL_B = JL_B - (JL_N * v1) * transpose(v1) * v2 + new_JL_column * transpose(H*m)/sqrt(p);
            JL_N = JL_N + new_JL_column * transpose(m/sqrt(p));



            % update solution
            v3 = G * a/sqrt(p);
            v4 = 1/(1+transpose(a) * G * a/p);
            G = G - v3 * transpose(v3) * v4;

            u = u + a * label/p;
            x = G * u;


        elseif isequal(method, 'row_sampling')
            a = transpose(A(d+1+t,:));
            label = b(d+1+t);
            m = [a; label];


            % leverage score sampling
            tau = transpose(m) * H * m;
            p = min(tau * samplingScale, 1); % sampling probability
            if rand > p
                continue;
            end

            count = count + 1;

            % woodbury
            v1 = H * m / sqrt(p);
            v2 = 1 / (1+ transpose(m)*H*m/p);
            H = H - v1 * transpose(v1) * v2;

            v3 = G * a / sqrt(p);
            v4 = 1 / (1 + transpose(a) * G * a/p);
            G = G - v3 * transpose(v3) * v4;

            u = u + a * label/p;
            x = G * u;
        elseif isequal(method, 'woodbury')
            a = transpose(A(d+1+t,:));
            label = b(d+1+t);


            % woodbury
            v1 = G * a;
            v2 = 1/(1+ transpose(a)*G*a);
            G = G - v1 * v2 * transpose(v1);

            % update solution
            u = u + a * label;
            x = G * u;
        elseif isequal(method, 'uniform')
            a = transpose(A(d+1+t,:));
            label = b(d+1+t);

            if rand > p
                continue;
            end

            count = count + 1;

            % woodbury
            v1 = G * a /sqrt(p);
            v2 = 1/(1+ transpose(a)*G*a / p);
            G = G - v1 * v2 * transpose(v1);

            % update solution
            u = u + a * label / p;
            x = G * u;


        end

    end
    
    tEnd = toc(tStart);
    
    
    
    
    total_time = total_time + tEnd;
    
    error = norm(A * x - b)^2 / (T+d+1);
    
    error_rate = error_rate + error/error_std;
    sample_rate = sample_rate + count /(T+d+1);
    

    
    
    
end

%----------------------------output statistics----------------------%


disp('Experiments finshed')

disp(['Dataset: ', dataset])

disp(['Algorithm: ', method])

disp(['Average total time is ', num2str(total_time/num_exp), 's'])

disp(['Average error_rate is ', num2str(error_rate/num_exp)])







    
    

