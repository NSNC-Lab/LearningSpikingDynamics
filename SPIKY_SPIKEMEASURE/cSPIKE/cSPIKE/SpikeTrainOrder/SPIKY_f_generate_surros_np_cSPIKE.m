function synf=f_generate_spike_order_surros(sto_profs,num_surros)

num_pairs=size(sto_profs,1);
num_trains=(1+sqrt(1+8*num_pairs))/2;
num_spikes=size(sto_profs,2);

[spike_indies,pair_indies,values]=find(sto_profs');
leader_pos=spike_indies(1:2:end);
follower_pos=spike_indies(2:2:end);
pairs=pair_indies(1:2:end);
coins=values(1:2:end);

num_coins=length(pairs);

[seconds,firsts]=find(triu(ones(num_trains),1)');
indies=[pairs firsts(pairs).*(coins==1)+seconds(pairs).*(coins==-1) seconds(pairs).*(coins==1)+firsts(pairs).*(coins==-1) leader_pos follower_pos]';
num_swaps=num_spikes;         % eliminate transients !!!!!
%disp(['First surrogate with long transient: ',num2str(num_swaps),' swaps --- All others without transient: ',num2str(round(num_swaps/2)),' swaps'])

synf=zeros(1,num_surros);
for suc=1:num_surros
    if suc==2
        num_swaps=round(num_swaps/2);
    end
    %disp('Start surro_MEX')
    [indies,~]=Spike_train_order_surro_MEX(indies,firsts,seconds,num_swaps);
    %disp('Stop surro_MEX')
    %disp([num2str(suc),')  Error_count: ',num2str(error_count)]);
    surro_sto_profs=zeros(size(sto_profs));
    for cc=1:num_coins
        surro_sto_profs(indies(1,cc),indies([4 5],cc))=(indies(2,cc)<indies(3,cc))-(indies(2,cc)>indies(3,cc))*ones(1,2);
    end
    
    surro_mat_entries=sum(surro_sto_profs,2)/2;   % to get from the spike train order profile to the spike order matrix
    surro_mat = tril(ones(num_trains),-1);
    surro_mat(~~surro_mat) = surro_mat_entries';
    surro_mat=surro_mat'-surro_mat;
    %disp('Start Spike_train_order_sim_ann_MEX')
    [~,synf(suc),~]=Spike_train_order_sim_ann_MEX(surro_mat);
    %disp('Stop Spike_train_order_sim_ann_MEX')
    if suc==num_surros
        %disp(['S#',num2str(suc),'  ',regexprep(num2str(synf*2/(num_trains-1)/sum(num_spikes),3),'   ',' ')])
    end
end

end


