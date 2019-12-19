function [ class, centroid ] = mykmeans( pixels, K )
%
% Your goal of this assignment is implementing your own K-means.
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

%   [class, centroid] = kmeans(pixels, K);

    MAX_ITERATION = 100;
    num_of_samps = size(pixels, 1);
##
##  % random initialization of centroids
##      centroid = randi(256, K, 3) - 1;
##
##      class = ones(num_of_samps, 1);
##      last_centroid = repmat(-1, K, 3);
##      
##      for iter = 1:MAX_ITERATION 
##          % assign samples
##          for i = 1:num_of_samps
##              % norm([255,255,255]) = 441.6730, 
##              % an impossibly large norm at initialzation
##              current_norm = 442;
##              for cla = 1:K
##                  this_norm = norm(pixels(i,:) - centroid(cla,:));
##                  if this_norm < current_norm
##                      current_norm = this_norm;
##                      class(i) = cla;
##                  end
##              end                
##          end
##          % update centroids
##          for cla = 1:K
##              logical_clustering = (class == cla)';
##              centroid(cla,:) = round(logical_clustering * pixels / (sum(logical_clustering)));
##          end
##          % if the centroids has not changed, stop iteration
##          if isequal(centroid, last_centroid)
##              break;
##          end
##          last_centroid = centroid;
##      end
##
##  end

