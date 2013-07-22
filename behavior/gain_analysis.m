datdir = '';
%%
[infile, datdir] = uigetfile([datdir '/.mat'],'multiselect','on');

if ~iscell(infile)
    if ischar(infile)
        infile = {infile};
    else
        datdir = '';
        return
    end
end
datdir
%%
clear groupStats
for i = 1:numel(infile)
    disp(['working on ' infile{i} ' file ' num2str(i) ' / ' num2str(numel(infile))]);
    try
        
    fname = infile{i}(1:regexp(infile{i},'\.')-1);
    groupStats(i).fname = fname;
    %%
    swim = load([datdir, infile{i}]);
    
    clear swimData
    
    swim = swim.swim;
    fig = {};
    POW = [];
    isHI = [];
    isLO = [];
    bouts = {};
    expParams  =nan(2);
    expStats = [];
    bouts = {};
    ds = 10;
    fs = 6000;
    L = length(swim.stimParam2);
    recDur = [1:ds:length(swim.fltCh2)]/(6000/ds);

    % No. of trials to warrant consideration of the data for gain analysis;
    trThresh = 5;
    % Minimum allowed ratio between mean of two channels for the lower of
    % the two to be analyzed
    trChan = 15;
    % Indicates quality of the channel
    qualChan = [1,1];
    chan = {'fltCh1','fltCh2'};
   
    figParams.pos = [10 300 1850 700];
    figParams.col = 'w';
    
    % currently (7/18/13) the backwards speed gui setting of the stimulus is -1200
    % * the values in stimparam1.
    velConv = -1200;
    % Print out some info about the experiment
    trLen = 19.9 * fs;        
    if numel(unique(swim.stimParam2(L/10:9*L/10))) == 2;
        expParams([1,2],1) = sort([mode(swim.stimParam2) mode(swim.stimParam2(swim.stimParam2~=mode(swim.stimParam2)))]);
        expParams(1,2) = velConv*mode(swim.stimParam1(swim.stimParam2 == expParams(1,1)));
        expParams(2,2) = velConv*mode(swim.stimParam1(swim.stimParam2 == expParams(2,1)));
    else
        % Need to use speed signal to extract trial structure
        expParams([1,2],1) = mode(swim.stimParam2);
        expParams([1,2],2) = velConv*sort([mode(swim.stimParam1) mode(swim.stimParam1(swim.stimParam1~=mode(swim.stimParam1)))]);
        % find the corners where the speed dropped
%         vel1 = find(swim.stimParam1 == expParams(1,2)/velConv);
%         vel2 = find(swim.stimParam1 == expParams(2,2)/velConv);
%         corners = {};
%         corners{1} = intersect(vel1+1,vel2);
%         if isempty(corners{1})
%             corners{1} = intersect(vel2+1,vel1);
%         end
%         trLen = median(diff(corners{1}))/2;

% find the other corners where the speed changed after being
        % constant for 3 samples
        corners{2} = find(swim.stimParam1(1:end-2) == swim.stimParam1(2:end-1) == swim.stimParam1(3:end));
        corners{2} = corners{2}(swim.stimParam1(corners{2}) ~= swim.stimParam1(corners{2}+1));
        
    end
    
    % Initial check for whether a channel sucks
    chanCheck = exp(abs(log(mean(swim.fltCh2)) - log(mean(swim.fltCh1))));
    groupStats(i).expParams = expParams;
    if chanCheck >15
        if mean(swim.fltCh2) > mean(swim.fltCh1)
            qualChan = [0 1];
        else
            qualChan = [1 0];
        end
    
    end
    
    try
        swimData(1)=kickassSwimDetect01(swim.ch1,swim.ch1);
        
    catch
        disp(['trouble with ' infile{i} ' on ch1'])
        expStats(1) = 0;
        qualChan(1) = 0;
    end
    
    try
        swimData(2)=kickassSwimDetect01(swim.ch2,swim.ch2);
        
    catch
        disp(['trouble with ' infile{i} ' on ch2'])
        expStats(2) = 0;
        qualChan(2) = 0;
    end
    
    gain1 = expParams(2,1);
    gain2 = expParams(1,1);
    
    condStr = {['g ' num2str(expParams(2,1)) ', v ' num2str(expParams(2,2))], ['g ' num2str(expParams(1,1)) ', v ' num2str(expParams(1,2))]};
    
     if abs(diff(expParams(:,1))) > .001
        hi = double(swim.stimParam2 == gain1);
        lo = double(swim.stimParam2 == gain2);
     else
         expParams
         disp('Moving to next file');
         continue
     end    
        
    if ishandle(1)
        set(0,'currentfigure',1);
    else
        figure(1)
    end
    
    set(gcf,'position',figParams.pos);
    set(gcf,'color',figParams.col);
    clf
    subplot(2,1,1); plot(recDur,swim.fltCh1(1:ds:end) .* hi(1:ds:end),'r'); hold on; plot(recDur,swim.fltCh1(1:ds:end).*lo(1:ds:end));
    axis tight
    set(gca,'ylim',quantile(swim.fltCh1,[.5, .9999]))
    title([fname ' fltCh1'],'interpreter','none')
    subplot(2,1,2); plot(recDur,swim.fltCh2(1:ds:end) .* hi(1:ds:end),'r'); hold on; plot(recDur,swim.fltCh2(1:ds:end).*lo(1:ds:end));
    axis tight
    legend(condStr)
    xlabel('Time (s)')
    set(gca,'ylim',quantile(swim.fltCh2,[.5, .9999]))
    title('fltCh2')
    
    print(gcf,[datdir fname 'fltChSummary'],'-dpng','-noui','-r200');
  
    if ishandle(2)
      set(0,'currentfigure',2);
      else
        figure(2)
    end
    set(gcf,'position',figParams.pos);
        set(gcf,'color',figParams.col);
    clf
    
    epochStarts = (abs(diff(swim.stimParam2))-min(abs(diff(swim.stimParam2))))./max(abs(diff(swim.stimParam2)));
    epochStarts = find(epochStarts) - 1;
    epochStops = epochStarts(2:end);
    epochStarts((epochStarts+trLen) > L) = [];  
    groupStats(i).epochs = epochStarts;
    if numel(epochStarts) < trThresh
        disp(['Not enough epochs in ' fname])
        continue
    end
    
    for p = find(qualChan);
        % Bin bouts by trial
        for g = 1:numel(epochStops)
            bouts{p}{g} = intersect(swimData(p).swimStartIndT,epochStarts(g):epochStops(g));
        end
        groupStats(i).bouts = bouts;
        if numel(swimData(p).swimStartIndT) > 10;
            for k=1:length(swimData(p).swimStartIndT)
                POW{p}(k)=sum(swim.(chan{p})(swimData(p).swimStartIndT(k):swimData(p).swimEndIndT(k)));
            end
            
            for k=1:length(swimData(p).swimStartIndT)
                isHI{p}(k)= (abs(swim.stimParam2(swimData(p).swimStartIndT(k))-gain1)<0.0001);
            end
            
            for k=1:length(swimData(p).swimStartIndT)
                isLO{p}(k)= (abs(swim.stimParam2(swimData(p).swimStartIndT(k))-gain2)<0.0001);
            end
            
            divs = min(POW{p}):2:max(POW{p});
            [a_hi b_hi] = hist(POW{p}(find(isHI{p}==1)),divs);
            [a_lo b_lo] = hist(POW{p}(find(isLO{p}==1)),divs);
            
            subplot(3,2,p)
            plot(divs,a_hi,'r--','linewidth',2)
            
            hold on
            plot(divs,a_lo,'--','linewidth',2)
            ylabel('Number of bouts')
            xlabel('Sum(swim signal) during bout')
            title({[fname ': ' chan{p}],'Power per bout'},'interp','none');
            legend(condStr);
            % Number of bouts per condition
            
            nBouts = cellfun(@numel,bouts{p});
                        
            % check what the first condition was
            cond1 = swim.stimParam2(epochStarts(1) + 10);
            if cond1 == gain1
                nboutsHi = nBouts(1:2:end);
                nboutsLo = nBouts(2:2:end);
            else
                nboutsLo = nBouts(1:2:end);
                nboutsHi = nBouts(2:2:end);
            end
            divs = (min(nBouts):1:max(nBouts))/(trLen/fs);
            [a_hi b_hi] = hist(nboutsHi/(trLen/fs),divs);
            [a_lo b_lo] = hist(nboutsLo/(trLen/fs),divs);
            subplot(3,2,p+2)
            bar(divs,a_hi,.9,'facecolor','r');
            hold on
            bar(divs,-a_lo,.9,'facecolor','b');
            axis tight
            hold off
             ylm = get(gca,'yticklabel');
            set(gca,'yticklabel',num2str(abs(str2num(ylm))));
            ylabel('Trials')
            xlabel('Bouts / s')
            title(['Bout freq.']);
            
            % Plot probability of swimming, power as a function of time
         
            subplot(3,2,p+4)

            normBouts = [];
            for r = 1:numel(bouts{p})
                if ~isempty(bouts{p}{r})
                normBouts(r) = bouts{p}{r}(1) - epochStarts(r);
                else
                    normBouts(r) = nan;
                end    
            end
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
        else
        disp(['Not enough bouts on ch ' num2str(p)])    
        end
            
    end
    catch 
        disp(['trouble with ' infile{i}]);
        continue
    end
end
disp('Done with main loop')
save([datdir 'groupStats'],'groupStats');
disp(['Done saving ' datdir 'groupStats']);



