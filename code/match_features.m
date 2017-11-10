% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the features as additional inputs.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2)
% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).
% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.
matches = [];
%pdist2 provides the pairwise distance between the features
dist = pdist2(features1, features2, 'euclidean');
% Sort each row of the distance matrix
[sort_dist, index] = sort(dist, 2);
%[idx,dist] = knnsearch(features1,features2);
nn_measure = (sort_dist(:,1)./sort_dist(:,2));

%applying the threshold
threshold = 0.90;
val_threshold(:) = nn_measure(nn_measure < threshold);
%finding the confidence vlaues
inverse_matrix = ones(size(val_threshold,1))*1;
confidences = inverse_matrix./val_threshold;
indices_feature(:)  = nn_measure<threshold;
%knn_measure_features(:) = knn_measure<threshold;
matches(:,1) = find(indices_feature);
matches(:,2) = index(indices_feature, 1);

%MD1 = KDTreeSearcher(features1);
%MD2 = KDTreeSearcher(features2);
%[idx1,dist] = knnsearch(features1,features2,'K',2,'NSMethod','kdtree');

%matches(:,1) = idx1(:,1);
%matches(:,2) = 1:size(features2,1);
%confidence_values = dist(:,1)./dist(:,2);
%[sort_confidence,indices] = sort(confidence_values);

%idx1 = idx1(:,1);
%matches(:,1) = idx1(indices);
%feat_new = features2(1:100);
% matches(:,2) = indices(feat_new,1);

%features2= features2(1:100,:);
% 
% for i = 1:size(features2,1) 
%     idx1 = knnsearch(MD1,features2(i,:));
%     matches(i,1) = idx1(:,1);
%     matches(i,2) = find(i);
% end
% idx2 = knnsearch(MD2,features1);
% idx1 = idx1(1:200,:);
% idx2 = idx2(1:200,:);
%size(idx1)
%matches(:,1) = idx1;
%[y,x]=find(features2);
%size(idx2)
%matches(:,2) = 0;
%matches(:,2) = idx2;


% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
matches = matches(ind,:);
%the top 100 matches
matches = matches(1:100,:);
%TRIED DOING REPMAT BUT IT TAKES O(m^2nlogn) COMPLEXITY. TIME TO RUN IS
%MORE. HENCE DISCARDED THIS METHOD.
% for i = 1:num_features1
%     distances = sqrt(sum((repmat(features1(i,:),num_features2,1)-features2).^2,2));    
%     %sorting according to the row
%     [dists,indices] = sort(distances,2);    
%     nnmeasure = dists(1)/dists(2);
%     matches(i,:) =  [i,indices(1)];        
%     confidences(i) = nnmeasure;
%     
%     
% end
% 
% [confidences, ind] = sort(confidences, 'descend');
% matches = matches(ind,:);
% Placeholder that you can delete. Random matches and confidences
%num_features = min(size(features1, 1), size(features2,1));
%matches = zeros(num_features, 2);
%matches(:,1) = randperm(num_features); 
%matches(:,2) = randperm(num_features);
%confidences = rand(num_features,1);

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
