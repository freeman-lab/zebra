% Script for converting an image stored as a text file back into an array

[inFile inPath] = uigetfile('*.txt');

fid = fopen([inPath inFile], 'r+');

tline = fgetl(fid);
count = 1;
data = [];

while ischar(tline)
    data(count,:) = str2num(tline);
    count = count + 1;
    tline = fgetl(fid);
end    
%%
% Size of the stack which we're trying to reconstruct from the text file
stackSize = [900, 1600, 11];
data = reshape(data,stackSize);
%%
% If the kmeans assigments picked out either cells or noise, the difference
% should be revealed in how spatially distributed each cluster is.
% As a proxy for a more serious measure of the density of the clusters,
% take the standard deviations of the x,y, and z coordinates of each
% cluster and multiply all three, then sort the clusters by this measure
clustDisp = [];
for i = 0:99
    [coordsy, coordsx] = find(data(:,:,:) == i);
    [coordsx ~] = ind2sub(stackSize(2:3),coordsx);
    clustDisp(i+1,1) = std(coordsy);
    clustDisp(i+1,2) = std(coordsx);
end
[~, sortClust] = sort(clustDisp(:,1).*clustDisp(:,2));
%%
% Plot a set of clusters attractively
toplot = 1:100;
for i = toplot
   pos = get(gcf,'position')
   close gcf
   imagesc(sum((ismember(data(:,:,:),sortClust(i)-1)),3))
   set(gca,'color','k')
   set(gcf,'color','k')
   set(gca,'units','n')
   set(gca,'position',[.01 .01 .95 .95])
   axis image
   a = title(['Cluster(s) ' num2str(i)]);
   set(a,'color','w')
   set(gcf,'position',pos)
   colormap(circshift(bone,[0 1]))
   colorbar
   waitforbuttonpress
end

%%
tempind = 2;
close all
for i = 1:11
    figure
    set(gcf,'position',kPos{i});
    set(gcf,'color','k')
    imagesc((data(:,:,i).*ismember(data(:,:,i),sortClust(tempind)-1)))
    set(gca,'units','n')
    set(gca,'position',[0 0 1 1])
    title([ 'Slice ' num2str(i)])
    colormap(hot)
end