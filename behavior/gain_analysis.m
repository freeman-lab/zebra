
%%
file = 'Z:\Davis\data\14july13\14july13_4.10chFlt';
swim = load_10chFltFunc(file);
swimData(1)=kickassSwimDetect01(swim.ch1,swim.ch1);
swimData(2)=kickassSwimDetect01(swim.ch2,swim.ch2);
%%
fig = {};
POW = [];
isHI = [];
isLO = [];
L = length(swim.stimParam2);
ds = 10;

gain1 = max(unique(swim.stimParam2(L/4:3*L/4)));
gain2 = min(unique(swim.stimParam2(L/4:3*L/4)));

hi = double(swim.stimParam2 == gain1);
lo = double(swim.stimParam2 == gain2);

fig{1} = figure(1); 
clf
subplot(2,1,1); plot(swim.fltCh1(1:ds:end) .* hi(1:ds:end),'r'); hold on; plot(swim.fltCh1(1:ds:end).*lo(1:ds:end)); 
axis tight
set(gca,'ylim',get(gca,'ylim')+[.01 0])
title('fltCh1')
subplot(2,1,2); plot(swim.fltCh2(1:ds:end) .* hi(1:ds:end),'r'); hold on; plot(swim.fltCh2(1:ds:end).*lo(1:ds:end)); 
axis tight
set(gca,'ylim',get(gca,'ylim')+[.01 0])
title('fltCh2')
set(gcf,'position',[10 300 1850 700]);
axmarg(gcf)

%%
fig{3} = figure(3);
clf
chans = {'fltCh1','fltCh2'};

epochStarts = (abs(diff(swim.stimParam2))-min(abs(diff(swim.stimParam2))))./max(abs(diff(swim.stimParam2)));
epochStarts = find(epochStarts) - 1;
nBouts = {};

for p = 1:2
    if numel(swimData(p).swimStartIndT) > 10;
        for i=1:length(swimData(p).swimStartIndT)
            POW{p}(i)=sum(swim.(chans{p})(swimData(p).swimStartIndT(i):swimData(p).swimEndIndT(i)));
        end
        
        for i=1:length(swimData(p).swimStartIndT)
            isHI{p}(i)= (abs(swim.stimParam2(swimData(p).swimStartIndT(i))-gain1)<0.0001);
        end
        
        for i=1:length(swimData(p).swimStartIndT)
            isLO{p}(i)= (abs(swim.stimParam2(swimData(p).swimStartIndT(i))-gain2)<0.0001);
        end
    
divs = min(POW{p}):2:max(POW{p});
[a_hi b_hi] = hist(POW{p}(find(isHI{p}==1)),divs);
[a_lo b_lo] = hist(POW{p}(find(isLO{p}==1)),divs);

subplot(2,2,p)
plot(divs,a_hi,'linewidth',2)
hold on
plot(divs,a_lo,'r','linewidth',2)
legend({'hi gain','lo gain'})
title(['Power: ' chans{p}]);

% Number of bouts per condition
for i = 1:(numel(epochStarts)-1)
    nBouts{p}(i) = numel(intersect(swimData(p).swimStartIndT,epochStarts(i):epochStarts(i+1)));
end

% check what the first condition was
cond1 = swim.stimParam2(epochStarts(1) + 10);
if cond1 == gain1
    nboutsHi = nBouts{p}(1:2:end);
    nboutsLo = nBouts{p}(2:2:end);
else
    nboutsLo = nBouts{p}(1:2:end);
    nboutsHi = nBouts{p}(2:2:end);
end
divs = min(nBouts{p}):1:max(nBouts{p});
[a_hi b_hi] = hist(nboutsHi,divs);
[a_lo b_lo] = hist(nboutsLo,divs);
subplot(2,2,p+2)
tmp = bar(divs,[a_hi;a_lo]',2);
axis tight
legend({'hi gain','lo gain'})
title(['Number of bouts: ' chans{p}]);
    end
    end
%%
% Assign each swim bout an index corresponding to when it occurred after a
% gain/speed transition, i.e. order bouts within conditions
boutOrder = zeros(length(swimData.swimStartIndT),1);
boutOrder(1) = 1;
prevGain = swim.stimParam2(swimData.swimStartIndT(1));
count = 1;
for i = 2:length(swimData.swimStartIndT)
    count = count + 1;
    curBout = swimData.swimStartIndT(i);
    
    if ~isequal(prevGain,swim.stimParam2(curBout))
        count = 1;
    end
    boutOrder(i) = count;
 
    prevGain = swim.stimParam2(curBout);
end
%%
% Calculate the average time of onset and power for the 1st, 2nd, 3rd bout in an
% epoch

% collect bouts occurring in the position specified by nBouts and divided
% by condition
nBouts = [1,2,3,4,5];
boutCollect = {};
for i = 1:numel(nBouts)
    tempBouts = find(boutOrder == nBouts(i));
    boutCollect{i,1} = tempBouts(swim.stimParam2(swimData.swimStartIndT(tempBouts)) == gain1); 
    boutCollect{i,2} = tempBouts(swim.stimParam2(swimData.swimStartIndT(tempBouts)) == gain2);
end
%%
% Now plot the timing of the first swim bouts
boutOnsets = {};
for i = 1:2
    for k = 1:numel(boutCollect{1,i})
        temp = swimData.swimStartIndT(boutCollect{1,i}(k)) - epochStarts;
        temp = temp(temp > 0);
        boutOnsets{i}(k) = min(temp)/6e3;
    end
end

figure
hist(boutOnsets')
legend({'slow -> fast','fast -> slow'})
title('Time before first swim bout after optic flow change');



    


