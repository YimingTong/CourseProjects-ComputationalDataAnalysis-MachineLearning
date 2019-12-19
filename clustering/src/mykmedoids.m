function [ class, centroid ] = mykmedoids( pixels, K )
%
% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with K rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

%	[class, centroid] = kmeans(pixels, K);

    MAX_ITERATION = 30;
% a compromise to bad optimization. It seems to be beyond my ability to
% realise a strict k_medoids with a resonable time/space complexity. Using
% double loops (since the alogrithm is of n^2 time complexity) seems to be
% bad optimized by MATLAB and will induce infinite running time. While
% turning into matrix computation will consume memory of 50GB
% approximately. I have seem the implementation of kmedoids from MATLAB
% itself. They also choose to ramdomly pick a subset of each cluster to be
% examined. Amazing that They can realise such a fast kmedoids function.
    EXAMINED_RATE = 0.001;
    num_of_samps = size(pixels, 1);

% random initialization of centroids within pixels
    centroid = zeros(K, 3);
    indexes = randperm(num_of_samps, K);
    for i = 1:K
        centroid(i, :) = pixels(indexes(i), :);
    end
    last_centroid = repmat(-1, K, 3);
    class = ones(num_of_samps, 1);    
    
    for iter = 1:MAX_ITERATION 
        % assign samples
        for i = 1:num_of_samps
            % norm([255,255,255]) = 441.6730, 
            % an impossibly large norm at initialzation
            current_norm = 442;
            for cla = 1:K
                this_norm = norm(pixels(i,:) - centroid(cla,:));
                if this_norm < current_norm
                    current_norm = this_norm;
                    class(i) = cla;
                end
            end                
        end
        % update centroids
        for cla = 1:K
            cls_size = sum(class == cla);
            examined_size = round(EXAMINED_RATE * cls_size);
            samps_this_cls = pixels(class == cla);
            examined_index = randperm(cls_size, examined_size);
            examined_samps = samps_this_cls(examined_index);
            current_cost = 0;
            for samp=samps_this_cls'
                current_cost = current_cost + norm(centroid(cla, :)-samp);
            end
            for temp_centroid = examined_samps'
                temp_cost = 0;
                for samp=samps_this_cls'
                    temp_cost = temp_cost + norm(temp_centroid - samp);
                end
                if temp_cost < current_cost
                    centroid(cla, :) = temp_centroid; 
                    break;
                end
            end 
        end
        if isequal(centroid, last_centroid)
            break;
        end
        last_centroid = centroid;
    end
end