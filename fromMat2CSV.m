%fromMat2Csv.m function translate the data file X_V.mat into 2 CSVs.
%
%The X_Y.mat contains 3 variables:
%   1. X: N-by-M data matrix. It contains N samples of M features
%   2. names: M-by-1 cell array. It contains the features names
%   3. Y: N-by-1 vector. It contains the targt value for each X's sample.
%
%The output files are:
%   1. data.csv: it contains the X matrix. Each column has as its first
%   cell the respective feature name.
%   2. target.csv: it contains the Y vector.

%% Prepare the workspace
close all;
clear all;
clc;
load(fullfile('data','X_Y'));

%% Write data.csv
fid = fopen(fullfile('data','data.csv'), 'w') ;
fprintf(fid, '%s,', names{1,1:end-1}) ;
fprintf(fid, '%s\n', names{1,end}) ;
fclose(fid) ;
dlmwrite(fullfile('data','data.csv'), names(2:end,:), '-append') ;
dlmwrite(fullfile('data','data.csv'), X, '-append') ;

%% Write target.csv
name = {'Y'};
fid = fopen(fullfile('data','target.csv'), 'w');
fprintf(fid, '%s\n', name{1,end}) ;
fclose(fid) ;
dlmwrite(fullfile('data','target.csv'),Y,'-append');
