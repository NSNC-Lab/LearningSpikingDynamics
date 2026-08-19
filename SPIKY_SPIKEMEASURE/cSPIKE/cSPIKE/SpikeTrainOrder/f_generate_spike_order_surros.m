function synf=f_generate_spike_order_surros(sto_profs,num_surros)

surro_plot=0;

num_pairs=size(sto_profs,1);
num_trains=(1+sqrt(1+8*num_pairs))/2;
num_all_spikes=size(sto_profs,2);

[spike_indies,pair_indies,values]=find(sto_profs');
leader_pos=spike_indies(1:2:end);
follower_pos=spike_indies(2:2:end);
pairs=pair_indies(1:2:end);
coins=values(1:2:end);

num_coins=length(pairs);

[seconds,firsts]=find(triu(ones(num_trains),1)');
indies=[pairs firsts(pairs).*(coins==1)+seconds(pairs).*(coins==-1) seconds(pairs).*(coins==1)+firsts(pairs).*(coins==-1) leader_pos follower_pos]';
num_swaps=num_all_spikes;         % eliminate transients !!!!!

%num_swaps=1; % #######

%disp(['First surrogate with long transient: ',num2str(num_swaps),' swaps --- All others without transient: ',num2str(round(num_swaps/2)),' swaps'])

synf=zeros(1,num_surros);
for suc=1:num_surros
    if suc==2
        num_swaps=round(num_swaps/2);
    end
    %num_swaps = 1;
    %disp('Start surro_MEX')
    
    if exist(['SPIKE_order_surro_MEX.',mexext],'file') && surro_plot==0   % #######
        [indies,error_count]=SPIKE_order_surro_MEX(indies,firsts,seconds,num_swaps);
        %disp([num2str(suc),')  Error_count: ',num2str(error_count)]);
        %indies_mat(suc+1,1:5,1:num_coins)=indies;
        %save long_test2 indies_mat
    else
        sc=1;
        error_count=0;
        while sc<=num_swaps
            brk=0;
            dummy=indies;
            
            coin=randi(num_coins,1);
            
            %coin=1;
            
            train1=indies(2,coin);   % important, don't use indies directly !
            train2=indies(3,coin);
            pos1=indies(4,coin);
            pos2=indies(5,coin);
            fi11=find(indies(4,:)==pos1);
            fi21=find(indies(5,:)==pos1);
            fi12=find(indies(4,:)==pos2);
            fi22=find(indies(5,:)==pos2);
            fiu=unique([fi11 fi21 fi12 fi22]);
            indies(2,fi11)=train2;
            indies(3,fi21)=train2;
            indies(2,fi12)=train1;
            indies(3,fi22)=train1;
            for fc=fiu
                new_trains=sort(indies([2 3],fc));                                   % switch train numbers
                indies(1,fc)=find(firsts==new_trains(1) & seconds==new_trains(2));   % update pairs
            end
            %indies
            for fc=fiu
                sed=setdiff(find(indies(1,:)==indies(1,fc)),fc);    % all other coincidences from that pair of spike trains
                for sedc=1:length(sed)
                    %sed(sedc)
                    %[fc indies(1,fc) sed(sedc) indies([4 5],sed(sedc))' indies([4 5],fc)']
                    if ~isempty(intersect(indies([4 5],sed(sedc)),indies([4 5],fc)))
                        error_count=error_count+1;
                        indies=dummy;
                        brk=1;
                        break
                    end
                end
                if brk==1
                    break
                end
            end
            if brk==1
                if error_count<=2*num_coins
                    continue
                else
                    sc=num_swaps;
                end
            end
            
            if surro_plot>0
                all_trains(indies([4 5],coin))=all_trains(indies([5 4],coin));    % ############### all_trains ?
            end
            if surro_plot==2
                surro_sto_profs=zeros(size(sto_profs));
                for cc=1:num_coins
                    surro_sto_profs(indies(1,cc),indies([4 5],cc))=(indies(2,cc)<indies(3,cc))-(indies(2,cc)>indies(3,cc))*ones(1,2);
                end
                surro_mat_entries=sum(surro_sto_profs,2)/2;
                surro_mat = tril(ones(num_trains),-1);
                surro_mat(~~surro_mat) = surro_mat_entries';
                surro_mat=surro_mat'-surro_mat;
                synfire=sum(sum(triu(surro_mat)));
                delta=synfire-old_synfire;
                overall_delta=synfire-ini_synfire;
                plot(-(sc-1)*(num_trains+1)-all_trains,'r*','LineWidth',2,'MarkerSize',ms)
                plot(indies([4 5],coin),-(sc-1)*(num_trains+1)-all_trains(indies([4 5],coin)),'bo','LineWidth',lw,'MarkerSize',ms+5)
                line(xl,(-(sc-1)*(num_trains+1))*ones(1,2),'Color','k','LineWidth',2,'LineStyle',':')
                text(xl(2)-0.15*(xl(2)-xl(1)),-(sc-1)*(num_trains+1)-num_trains/2-0.5,num2str(synfire),'Color','k','FontWeight','bold','FontSize',fs)
                text(xl(2)-0.1*(xl(2)-xl(1)),-(sc-1)*(num_trains+1)-num_trains/2-0.5,num2str(delta),'Color','k','FontWeight','bold','FontSize',fs)
                text(xl(2)-0.05*(xl(2)-xl(1)),-(sc-1)*(num_trains+1)-num_trains/2-0.5,num2str(overall_delta),'Color','k','FontWeight','bold','FontSize',fs)
                old_synfire=synfire;
            end
            sc=sc+1;
            %if mod(sc,round(num_swaps/5))==0 [suc sc] end
        end
    end

    
    %[indies,~]=Spike_train_order_surro_MEX(indies,firsts,seconds,num_swaps)
    
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
    
    surro_mat=[0 3 3 -1 2 -1; -3 0 6 -1 -1 1; -3 -6 0 3 1 -6; 1 1 -3 0 4 -2; -2 1 -1 -4 0 3; 1 -1 6 2 -3 0];
    
    [a,synf(suc),b]=Spike_train_order_sim_ann_MEX(surro_mat)
    %disp('Stop Spike_train_order_sim_ann_MEX')
    if suc==num_surros
        %disp(['S#',num2str(suc),'  ',regexprep(num2str(synf*2/(num_trains-1)/sum(num_all_spikes),3),'   ',' ')])
    end
end

end


