function data = kickAssSwimDetect01(ch1, ch2)

% Output:
%
% data.swimStartIndT - time indices of where swims start
% data.swimEndIndT - ... where swims end
% data.swimStartT - time series which is 1 where swims start
% data.swimEndT - ... 1 where swims end
% data.burstIndT1 - time indices of burst locations on channel 1
% data.burstIntT2 - ... on channel 2
% data.fltCh1 - processed channel 1 (~ standard deviations)
% data.fltCh2 - ... processed channel 2
% data.ch1 - raw channel 1
% data.ch2 - ... channel 2
% data.burstBothT - time indices of burst locations on either channel
%                     1 for ch2 burt, 2 for ch2 burst
% data.swimStartIndB - indices of data.burstBothT where swims start
% data.swimEndIndB - ... where swims end
% data.th1 - threshold for channel 1 used to extract bursts
% data.th2 - ... for channel 2
    

fprintf('\nProcessing channel data\n')
ker = exp(-(-60:60).^2/(2*20^2));
ker = ker / sum(ker);

smch1 = conv(ch1,ker);
smch1 = smch1(61:end-60);
pow1 = (ch1 - smch1).^2;
fltCh1 = conv(pow1,ker);
fltCh1 = fltCh1(61:end-60);

smch2 = conv(ch2,ker);
smch2 = smch2(61:end-60);
pow2 = (ch2 - smch2).^2;
fltCh2 = conv(pow2,ker);
fltCh2 = fltCh2(61:end-60);

fprintf('\nExtracting peaks\n')
aa1 = diff(fltCh1);
peaksT1 = (aa1(1:end-1) > 0) .* (aa1(2:end) < 0);
peaksIndT1 = find(peaksT1);

aa2 = diff(fltCh2);
peaksT2 = (aa2(1:end-1) > 0) .* (aa2(2:end) < 0);
peaksIndT2 = find(peaksT2);


fprintf('\nFinding thresholds\n')
x_ = 0:0.00001:0.1;
th1 = zeros(size(fltCh1));
th2 = zeros(size(fltCh2));

d_ = 6000*60*5;   % 5 minutes threshold window

j=0;
for i = 1:d_:length(fltCh1)-d_
    peaksIndT1_ = find(peaksT1(1:i+d_-1));
    peaksIndT2_ = find(peaksT2(1:i+d_-1));

    a1 = hist(fltCh1(peaksIndT1_), x_);
    a2 = hist(fltCh2(peaksIndT2_), x_);
    
    mx1 = min(find(a1 == max(a1)));
    mn1 = max(find(a1(1:mx1) < a1(mx1)/200));
    mx2 = min(find(a2 == max(a2)));
    mn2 = max(find(a2(1:mx2) < a2(mx2)/200));
    
%    th1(i:i+d_) = x_(mx1) + 1.95*(x_(mx1)-x_(mn1));
%    th2(i:i+d_) = x_(mx2) + 1.95*(x_(mx2)-x_(mn2));
    th1(i:i+d_) = x_(mx1) + 1.6*(x_(mx1)-x_(mn1));
    th2(i:i+d_) = x_(mx2) + 1.6*(x_(mx2)-x_(mn2));
end

th1(i+d_+1:end) = th1(i+d_);
th2(i+d_+1:end) = th2(i+d_);

fprintf('\nAssigning bursts and swims\n');
burstIndT1 = peaksIndT1(find(fltCh1(peaksIndT1) > th1(peaksIndT1)));
burstT1=zeros(size(fltCh1));
burstT1(burstIndT1)=1;

burstIndT2 = peaksIndT2(find(fltCh2(peaksIndT2) > th2(peaksIndT2)));
burstT2=zeros(size(fltCh2));
burstT2(burstIndT2)=1;
 
burstBothT = zeros(size(fltCh1));
burstBothT(burstIndT1) = 1;
burstBothT(burstIndT2) = 2;

burstBothIndT = find(burstBothT > 0);

interSwims = diff(burstBothIndT);

swimEndIndB = [find(interSwims > 100/1000*6000) ;  length(burstBothIndT)];
swimStartIndB = [1;  swimEndIndB(1:end-1) + 1];

nonSuperShort = find(swimEndIndB ~= swimStartIndB);
swimEndIndB = swimEndIndB(nonSuperShort);
swimStartIndB = swimStartIndB(nonSuperShort);

% swimStartIndB is an index for burstBothIndT
% burstBothIndT is an idex for time

swimStartIndT = burstBothIndT(swimStartIndB);
swimStartT = zeros(size(fltCh1));
swimStartT(swimStartIndT) = 1;

swimEndIndT = burstBothIndT(swimEndIndB);
swimEndT = zeros(size(fltCh1));
swimEndT(swimEndIndT) = 1;


data.burstBothT = burstBothT;
    clear burtsBothT;
data.burstBothIndT = burstBothIndT;
    clear burtsBothIndT;
data.burstIndT1 = burstIndT1;
    clear burstIndT1;
data.burstIndT2 = burstIndT2;
    clear burstIntT2;
data.swimStartIndB = swimStartIndB;
    clear swimStartIndB;
data.swimEndIndB = swimEndIndB;
    clear swimEndIndB;
data.swimStartIndT = swimStartIndT;
    clear swimStartIndT;
data.swimEndIndT = swimEndIndT;
    clear swimEndIndT;
data.swimStartT = swimStartT;
    clear swimStartT;
data.swimEndT = swimEndT;
    clear swimEndT;
data.fltCh1 = fltCh1;
    clear fltCh1;
data.fltCh2 = fltCh2;
    clear fltCh2;
data.ch1 = ch1;
    clear ch1;
data.ch2 = ch2;
    clear ch2;
data.th1 = th1;
    clear th1;
data.th2 = th2;
    clear th2;
    


