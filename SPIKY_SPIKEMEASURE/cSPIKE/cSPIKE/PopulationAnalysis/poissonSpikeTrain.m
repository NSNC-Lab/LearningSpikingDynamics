function spikeTimes = poissonSpikeTrain( time, rate,refract )
   
if nargin < 3
    refract = 0;
end

ISIs = f_poisson(ceil(time*rate*2),rate,refract);
spikeTimes = cumsum(ISIs);
spikeTimes = spikeTimes(spikeTimes<time);
end

