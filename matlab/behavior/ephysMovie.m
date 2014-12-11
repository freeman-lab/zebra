%% Load a series of images with dimensions [x,y,t] as well as 10chFlt file corresponding to that series and display together
%% Specify image file and ephys file here
imFile = '/Users/bennettd/Desktop/data/GFAP_GC6F_9DPF_ASS_1_20141123_012256.mat';
epFile = '/Users/bennettd/Desktop/data/11232014_GFAP_GC6F_9DPF_ASS_1.10chFlt';
%% load images
load(imFile);
%% load ephys
fid = fopen(epFile);
epDat = fread(fid,'float');
fclose(fid);
%% get ephys channels
ch1 = epDat(1:10:end);
ch2 = epDat(2:10:end);
frameCh = epDat(3:10:end);
velCh = epDat(9:10:end);
condCh = epDat(7:10:end);
%% Filter ephys
fltch1 = sqrt(smooth((ch1 - smooth(ch1,100)).^2,100));
fltch2 = sqrt(smooth((ch2 - smooth(ch2,100)).^2,100));
%% get frame onset times

thrMag = 3.8;
thrDur = 10;
stackInits = find(frameCh > thrMag);
initDiffs = find(diff(stackInits) > 1);
initDiffs = [1; initDiffs];    
stackInits = stackInits(initDiffs);
keepers = [find(diff(stackInits) > thrDur); length(stackInits)-1];
stackInits = stackInits(keepers);
imPeriod = round(median(diff(stackInits)));
%%
figure(999)
ax = [];
ax(1) = subplot(2,2,1:2);
ax(2) = subplot(2,2,3:4);

marg = [.03,.03];
set(999,'currentaxes',ax(1))
set(gca,'position',[marg(1), .3 + marg(2), 1-marg(1), .7-marg(2)])
set(999,'currentaxes',ax(2))
set(gca,'position',[marg(1), .01, 1-marg(1), .3-marg(2)])

clim = [0,2000];
colormap gray
epWind1 = [0,imPeriod];
epWind2 = [12000,12000];
epYLim = [-max(fltch1), max(fltch2)]/2;
dt = .3;

% whole recording
 tRange = 1:size(mxProj,3);

for n=tRange            
    set(999,'currentaxes',ax(1))
    tmpx = round(get(gca,'xlim'));
    tmpy = round(get(gca,'ylim'));
    
    % draw image
    imagesc(mxProj(:,:,n),clim)           
    set(gca,'ydir','normal')
    axis off
    axis image
    
    % allow zooming during video
    if n > tRange(1)
        set(gca,'xlim',tmpx)
        set(gca,'ylim',tmpy)
    end
    
    % draw ephys
    set(999,'currentaxes',ax(2))
    if n > ceil(epWind2(1) / imPeriod)
        cla
        plot(fltch1(stackInits(n) - epWind2(1) : stackInits(n) + epWind2(2)),'k','linewidth',2);
        hold on
        plot(-fltch2(stackInits(n) - epWind2(1) : stackInits(n) + epWind2(2)),'b','linewidth',2);
        plot(.05 + 10*velCh(stackInits(n) - epWind2(1) : stackInits(n) + epWind2(2)),'r','linewidth',2);
        axis tight
        line([epWind2(1),epWind2(1)],[0,1],'linewidth',3) 
        set(gca,'ylim',epYLim)
        axis off
        title(['Frame ' num2str(n)])
    end
    
    pause(dt)
end
