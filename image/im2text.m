% Take a series of JPEG2000 images representing a volume imaged over time
% and sticks them all in a text file with dimensions voxels X time points
% Image files must be formatted thus: name_T1_slice2.jp2


inPath = uigetdir('Z:\Shared\Data\data_fish7_sharing_sample\data_for_sharing_01\12-10-05\Dre_L1_HuCGCaMP5_0_20121005_154312.corrected.processed\dff_aligned','Select directory of JPEG200 files');
fileName = inputdlg('Pick a name for output file');
% outPath = uigetdir('Pick a folder to save in');

% Specify a set of pixels to grab as a mask
% outmask = foo

slices = 1:maxZ;
times = 1:maxTime;

%%

allFiles = cellstr(ls(inPath));
% Trim off the '.' and '..' lines at the top of the ls output
allFiles = allFiles(3:end);

% Strike out the max files
maxFiles = ~cellfun(@isempty, regexp(allFiles,'max'));
allFiles(maxFiles) = [];

% Figure out the dimensions of the image by loading a single file 
imIn = imread([inPath filesep allFiles{1}]);
imSize = size(imIn);
% Now figure out Z and T axes
timeInds = regexp(allFiles,'T[0-9]*','match');
timeInds = cellfun(@cell2mat,timeInds,'un',0);
timeInds = cellfun(@str2num, (cellfun(@cell2mat,regexp(timeInds,'[0-9]*','match'),'un',0)));
maxTime = max(timeInds);

zInds = regexp(allFiles,'_slice[0-9]*','match');
zInds = cellfun(@cell2mat,zInds,'un',0);
zInds = cellfun(@str2num, (cellfun(@cell2mat,regexp(zInds,'[0-9]*','match'),'un',0)));
maxZ = max(zInds);

% write to text file where each row is a time point
f = fopen([inPath filesep fileName{1} '_tx.txt'],'w');
disp('Loading data...')
for it = times
    tmp = zeros(imSize(1),imSize(2),maxZ);
    for is = slices
        foo = imread(fullfile(inPath,['/dff_aligned_T' num2str(it) '_slice' num2str(is) '.jp2']));
        % Apply mask 
        tmp(:,:,is) = (double(foo(imSize(1),imSize(2))) - 15000)/5000;
    end
    tmp = tmp(:)';
    fmt = repmat('%.4f ',1,length(tmp)-1);
    fprintf(f,[fmt,'%.4f\n'],tmp);
end
fclose(f);
disp('Done writing file')