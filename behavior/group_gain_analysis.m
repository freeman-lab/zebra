%% Group swim data analysis
% determine which files to analyze as a group

% look at timing of first swim bouts by condition by fish
% first look at the fish from june 30 in the proportional conditions
% toProc = {'06302013_f1_21.10chFlt','06302013_f1_18.10chFlt',...
%     '0.010/1.000 0.040/4.000','06302013_f1_12.10chFlt','06302013_f1_9.10chFlt'


%%

% find all the recordings with fixed gain/vel ratio
datdir = 'Z:\Davis\data\ephys\pooled\';
load([datdir,'groupStats'])

%%
toProc = [];
for i = 1:numel(groupStats)
    if ~isempty(groupStats(i).bouts)
        if diff(groupStats(i).expParams(1,:)./groupStats(i).expParams(2,:)) == 0
            if diff(groupStats(i).expParams(:,1)) > 0;
                toProc(end+1) = i;
                groupStats(i).expParams
            end
        end
    end
end
%%
fs =6000;
trLen = 19.7 * fs;
clear swimData
bouts = {};
normBouts = {};
qualChan = ones(numel(toProc),2);
epochStarts = {};
epochStops = {};
for i = 1:numel(toProc)
    i
    load([datdir groupStats(toProc(i)).fname '.mat']);
    clear swimData
    chanCheck = exp(abs(log(mean(swim.fltCh2)) - log(mean(swim.fltCh1))));
    if chanCheck >15
        if mean(swim.fltCh2) > mean(swim.fltCh1)
            qualChan(i,:) = [0 1];
        else
            qualChan(i,:) = [1 0];
        end
    end
    try
        swimData(i,1) = kickassSwimDetect01(swim.ch1,swim.ch1);
    catch
        disp(['trouble with swim.ch1']);
        qualChan(i,1) = 0;
    end
    try
        swimData(i,2) = kickassSwimDetect01(swim.ch2,swim.ch2);
    catch
        disp(['trouble with swim.ch1']);
        qualChan(i,2) = 0;
    end

    groupStats(toProc(i)).fname
    qualChan(i,:)
    epochStarts{i} = (abs(diff(swim.stimParam2))-min(abs(diff(swim.stimParam2))))./max(abs(diff(swim.stimParam2)));
    epochStarts{i} = find(epochStarts{i}) - 1;
    epochStops{i} = epochStarts{i}(2:end);
    epochStarts{i}((epochStarts{i}+trLen) > length(swim.stimParam2)) = [];
    epochs = swim.stimParam2(epochStarts{i} + 10);
    for p = 1:find(squeeze(qualChan(i,:)))
        for g = 1:numel(epochStops{i})
            bouts{i}{p}{g} = intersect(swimData(i,p).swimStartIndT,epochStarts{i}(g):epochStops{i}(g));
        end
        
        for r = 1:numel(bouts{i}{p})
            if ~isempty(bouts{i}{p}{r})
                normBouts{i}{p}{r} = bouts{i}{p}{r}(1) - epochStarts{i}(r);
            else
                normBouts{i}{p}{r} = nan;
            end
        end
        
        
    end
end

%%

% the idea: plot initial bout times color-coded by the offset velocity at
% the time of swimming


divs = linspace(0,trLen/fs,400);
toBar = [];
if cond1 == gain1
    toBar(:,1) = hist(normBouts(1:2:end)/fs,divs);
    toBar(:,2) = -hist(normBouts(2:2:end)/fs,divs);
else
    toBar(:,1) = hist(normBouts(2:2:end)/fs,divs);
    toBar(:,2) = -hist(normBouts(1:2:end)/fs,divs);
end
plot(divs,toBar(:,1),'r--','linewidth',2)
hold on
plot(divs,toBar(:,2),'--','linewidth',2)
title('Timing of first bouts')
xlabel('Time after trial onset (s)');
axis tight;
ylm = get(gca,'yticklabel');
set(gca,'yticklabel',num2str(abs(str2num(ylm))));
set(gca,'xlim',[0 (trLen/fs)/4]);
hold off
print(gcf,[datdir fname 'BoutSummary'],'-dpng','-noui','-r200');



