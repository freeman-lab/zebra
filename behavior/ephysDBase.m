datdir = '';
%%
[infile, datdir] = uigetfile([datdir '/.10chflt'],'multiselect','on');

if ~iscell(infile)
    if ischar(infile)
        infile = {infile};
    else
        datdir = '';
        return
    end
end
datdir
outfile = fopen([datdir datdir(1+max(regexp(datdir(1:end-1),'\')):end-1) '_expData.txt'],'w');
for i = 1:numel(infile)
    fname = infile{i}(1:regexp(infile{i},'\.')-1);
    swim = load_10chFltFunc([datdir infile{i}]);
    clear swimData;
    expParams  =nan(2);
    expStats = [];
    % currently (7/18/13) the backwards speed gui setting of the stimulus is -1200
    % * the values in stimparam1.
    velConv = -1200;
    % Print out some info about the experiment
    
    if numel(unique(swim.stimParam2)) > 1
        expParams([1,2],1) = sort([mode(swim.stimParam2) mode(swim.stimParam2(swim.stimParam2~=mode(swim.stimParam2)))]);
        expParams(1,2) = velConv*mode(swim.stimParam1(swim.stimParam2 == expParams(1,1)));
        expParams(2,2) = velConv*mode(swim.stimParam1(swim.stimParam2 == expParams(2,1)));
    else
        expParams([1,2],1) = swim.stimParam2(1);
        expParams([1,2],2) = velConv*sort([mode(swim.stimParam1) mode(swim.stimParam1(swim.stimParam1~=mode(swim.stimParam1)))]);
    end
   
    try
        swimData(1)=kickassSwimDetect01(swim.ch1,swim.ch1);
    catch
        disp(['trouble with ' infile{i} 'ch1'])
        swimData(1).swimStartIndT = 0;
    end
    try
        swimData(2)=kickassSwimDetect01(swim.ch2,swim.ch2);
    catch
        disp(['trouble with ' infile{i} ' ch2'])
        swimData(2).swimStartIndT = 0;
    end
    % calculate the mean swimming rate on each channel
    expStats(1) = numel(swimData(1).swimStartIndT)/(length(swim.stimParam1)/6000);
    expStats(2) = numel(swimData(2).swimStartIndT)/(length(swim.stimParam1)/6000);
    
    formatSpec = '%s, %.3f, %.3f/%.3f %.3f/%.3f, %.3f, %.3f \n';
    fprintf(outfile,formatSpec,infile{i},length(swim.stimParam1)/6000/60,...
        expParams(1,1), expParams(1,2),expParams(2,1),expParams(2,2),...
        expStats(1), expStats(2))
    
    save([datdir fname '_preProc'],'swimData','swim','expParams','expStats')
end
fclose(outfile);