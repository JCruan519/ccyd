function [q1,q2,q3,y_raw] = test(path)

sample = [1,inf];

[y_raw,Fs] = audioread(path , sample);
%loading audio file
[Q1,Q2,Q3] = AudioDivisionFunc(y_raw,Fs);
q1 = Q1
q2 = Q2
q3 = Q3
y_raw = y_raw
hold on
plot(y_raw,'color','B');
plot(Q1,0,'*','color','R');
plot(Q2,0,'*','color','R');
plot(Q3,0,'*','color','R');
hold off

end
