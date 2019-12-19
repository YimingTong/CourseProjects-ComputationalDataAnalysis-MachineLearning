function  homework2_PCA()
    
    % Read the data
    Name = dir();  % Get all the files' name in the document
    Data = [];
    for i = 1:length(Name)
        if length(Name(i).name)>11
            if strcmp(Name(i).name(1:9),'subject14') & (Name(i).name(11)~='t') % The subject.*** name we need
                image = imread(Name(i).name);% Get the image
                image = imresize(image,0.25);
                rows = size(image, 1);
                cols = size(image, 2);% The rows and columns of the image
                pixels = double(reshape(image(:,:,1),[rows*cols,1]));
                Data = [Data, pixels];
            end
        end
    end
    
    
        %Question 1
        
        %Mean 
        M = mean(Data,2);
        
        Mean_face = uint8(reshape(M,[rows,cols]));
        imagesc(Mean_face);
        hold off;
        
        
        colormap gray
        %The first 6 Eigen Faces
        D = Data-repmat(M,1,10);
        m = size(D,2);
        C = D * D' ./ m; 
        [W, S] = eigs(C, 6);

        
        for i = 1:6
            subplot(2,3,i);
            imagesc(reshape(W(:,i),rows,cols));
        end
       

        %Question 2
        % Use the first component
        U = W(:,1);
        
        % Get the test file for subject14
        Test= imread('subject14.test.gif');% Get the test image
        Test = imresize(Test,0.25);
        T = double(reshape(Test(:,:,1),[rows*cols,1]));
        score_1 = T'*U;
        display(score_1);
        
        % Get another different file!
        Test= imread('subject01.gif');% Get the test image
        Test = imresize(Test,0.25);
        T = double(reshape(Test(:,:,1),[rows*cols,1]));
        score_2 = T'*U;
        display(score_2);
        
        % Get the test file for subject14
        Test= imread('subject14.test.gif');% Get the test image
        Test = imresize(Test,0.25);
        T = double(reshape(Test(:,:,1),[rows*cols,1]));
        score_1 = T'*U;
        display(score_1);
        
        % Get another different file!
        Test= imread('subject01.gif');% Get the test image
        Test = imresize(Test,0.25);
        T = double(reshape(Test(:,:,1),[rows*cols,1]));
        score_2 = T'*U;
        display(score_2);
        
        % Get the test file for subject14
        Test= imread('subject01.happy.gif');% Get the test image
        Test = imresize(Test,0.25);
        T = double(reshape(Test(:,:,1),[rows*cols,1]));
        score_3 = T'*U;
        display(score_3);
        
        % Get another different file!
        Test= imread('subject01.sad.gif');% Get the test image
        Test = imresize(Test,0.25);
        T = double(reshape(Test(:,:,1),[rows*cols,1]));
        score_4 = T'*U;
        display(score_4);
        

        
end
