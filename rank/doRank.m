function doRank()
    %KxK matrix
    K = 10;
    %how many samples
    perRankTr = 100000;
    perRankTe = 1000;

    fprintf('Generating Samples\n');
    fprintf('This may take a while\n');
    setup(K,perRankTr,perRankTe);
    fprintf('Learning\n');
    fprintf('This will take as much as an hour\n');
    runsolver(K,perRankTr,perRankTe);
    fprintf('Testing\n');
    runtest(K,perRankTr,perRankTe);

    ye = dlmread('ye.txt');
    yep = dlmread('yep.txt');
    [~,yh] = max(yep,[],2); yh = yh - 1;

    fprintf('Accuracy: %.2f\n',100*mean(yh==ye)); 
end

function runsolver(K,perRankTr,perRankTe)
    system(sprintf('../caffe64 train ranknet.txt rank.bin optimsetting.txt %d Xr.txt yr.txt > /dev/null',perRankTr*K));
end

function runtest(K,perRankTr,perRankTe)
    system(sprintf('../caffe64 test ranknet.txt rank.bin %d Xe.txt yep.txt > /dev/null',perRankTe*K));
end

function setup(K,perRankTr,perRankTe)

    Xr = zeros(K*perRankTr,K*K);
    Xe = zeros(K*perRankTe,K*K);
    yr = zeros(K*perRankTr,1);
    ye = zeros(K*perRankTe,1);

    for i=1:K
        startR = (i-1)*perRankTr; endR = i*perRankTr;
        startE = (i-1)*perRankTe; endE = i*perRankTe;

        yr(startR+1:endR) = i; ye(startE+1:endE) = i;

        for j=1:perRankTr,
            M = genM(K,i); Xr(startR+j,:) = M(:);
        end

        for j=1:perRankTe,
            M = genM(K,i); Xe(startE+j,:) = M(:);
        end
    end

    dlmwrite('Xr.txt',Xr,' ');
    dlmwrite('Xe.txt',Xe,' ');
    dlmwrite('yr.txt',yr-1,' ');
    dlmwrite('ye.txt',ye-1,' ');
end

function M = genM(K,r)
    %Generate a KxK matrix with rank r
    [U,S,V] = svd(rand(K,K));
    So = S;
    %start off by forcing K
    S = diag(S + 1e-4*eye(K));
    %cut to r
    if r < K, S(r+1:end) = 0; end
    M = U*diag(S)*V;
    M = M ./ sum(M(:));
end
