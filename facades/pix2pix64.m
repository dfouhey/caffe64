%David Fouhey
%Caffe64 with MATLAB

function pix2pix64(doTrain,doTest,direction)
    trainSrc = 'facades/train/';
    testSrc = 'facades/test/';
    resultTarget = 'facades/results/';
    resultTargetDirection = sprintf('%s/direction_%d/',resultTarget,direction);

    if ~exist(resultTarget), mkdir(resultTarget); end
    if ~exist(resultTargetDirection), mkdir(resultTargetDirection); end

    jpgsTrain = dirCell(trainSrc);
    jpgsTrain = jpgsTrain(1:min(1000,numel(jpgsTrain)));
    jpgsTest = dirCell(testSrc);

    subsample = 1000;

    halfW = 8;
    fullW = halfW*2+1;
    featureSize = fullW*fullW;

    if doTrain

        X = zeros(numel(jpgsTrain)*subsample,featureSize*3);
        y = zeros(numel(jpgsTrain)*subsample,3);

        for i=1:numel(jpgsTrain)
            fprintf('Handling %d/%d\n',i,numel(jpgsTrain));
            I = imread(sprintf('%s/%d.jpg',trainSrc,i)); 
            I = imresize(I,[128,256]);

            if direction, Iout = I(:,129:256,:); Iin = I(:,1:128,:);
            else, Iin = I(:,129:256,:); Iout = I(:,1:128,:);
            end

            target = reshape(Iout,[],3);
            k = randperm(size(target,1),subsample);
            blockStart = (i-1)*subsample+1; blockEnd = i*subsample;

            y(blockStart:blockEnd,:) = target(k,:);

            for c=1:3
                Iinp = padarray(Iin(:,:,c),[halfW,halfW]);
                Iinc = im2col(Iinp,[fullW,fullW],'sliding')';
                featStart = (c-1)*featureSize+1;
                featEnd = c*featureSize;
                X(blockStart:blockEnd,featStart:featEnd) = Iinc(k,:);
            end
        end

        X = (X-128)/128;
        y = y ./ 255;

        fprintf('Doing a learning problem with %d instances with %d features\n',size(X,1),size(X,2));
        Xf = sprintf('./X_%d.txt',direction);
        dlmwrite(Xf,X,' ');
        for c=1:3
            yf = sprintf('./y_%d_%d.txt',c,direction);
            fprintf('Learning channel %d\n',c);
            dlmwrite(yf,y(:,c));
            com = sprintf('../caffe64 train networkRegDeep.txt facades_%d_%d.bin optimsettingDeep.txt %d %s %s > /dev/null',direction,c,size(X,1),Xf,yf);
            system(com);
            delete(yf);
        end
        delete(Xf);
    end

    if doTest
        numel(jpgsTest)
        for i=1:numel(jpgsTest)
            fprintf('Testing %d/%d\n',i,numel(jpgsTest));
            I = imread(sprintf('%s/%d.jpg',testSrc,i)); 
            I = imresize(I,[128,256]);

            if direction, Iout = I(:,129:256,:); Iin = I(:,1:128,:);
            else, Iin = I(:,129:256,:); Iout = I(:,1:128,:);
            end

            Xs = {};
            for c=1:3
                Iinp = padarray(Iin(:,:,c),[halfW,halfW]);
                Iinc = im2col(Iinp,[fullW,fullW],'sliding')';
                Xs{end+1} = double(Iinc);
            end
            X = cat(2,Xs{:});

            X = (X-128)/128.0;
           
            Xf = sprintf('./Xe_%d.txt',direction);
            yf = sprintf('./ye_%d.txt',direction);

            dlmwrite(Xf,X,' ');

            Is = {};
            for c=1:3
                system(sprintf('../caffe64 test networkRegDeep.txt facades_%d_%d.bin %d %s %s > /dev/null',direction,c,size(X,1),Xf,yf));
                ye = dlmread(yf)*255;
                I = reshape(ye,size(Iin(:,:,1)));
                Is{c} = uint8(I);
            end
            Is = cat(3,Is{:});
            result = cat(2,Iin,Is,Iout);
            imwrite(result,sprintf('%s/%06d.png',resultTargetDirection,i));
            delete(Xf);
            delete(yf);
        end
    end
end

function S = dirCell(t)
    files = dir(t);
    S = {};
    for i=1:numel(files)
        n = files(i).name;
        if n(1) ~= '.', S{end+1} = n; end
    end
end
