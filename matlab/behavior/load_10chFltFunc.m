
function out =  load_10chFltFunc(infiles)

h = fopen(infiles);
X = fread(h,'float');
fclose(h);

out.ch1 = X(1:10:end);
out.ch2 = X(2:10:end);
out.ch3 = X(3:10:end);
out.ch1aux = X(4:10:end);
out.ch2aux = X(5:10:end);
out.ch_12lowPass = X(6:10:end);
out.digDat = X(7:10:end);
out.stimID = X(8:10:end);
out.stimParam1 = X(9:10:end);
out.stimParam2 = X(10:10:end);

clear X
out.fltCh1aux = sqrt(smooth((out.ch1aux-smooth(out.ch1aux,100)).^2,100));
out.fltCh2aux = sqrt(smooth((out.ch2aux-smooth(out.ch2aux,100)).^2,100));


out.fltCh1 = sqrt(smooth((out.ch1-smooth(out.ch1,100)).^2,100));
out.fltCh2 = sqrt(smooth((out.ch2-smooth(out.ch2,100)).^2,100));



