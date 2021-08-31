%% this function is a 3-cluter Normal mixture model to simulate Daily Precipitation/Temperature
%% distribution using the Gibbs Sampler:
%% 1st cluster: light precipitation/cold temperature
%% 2nd cluster: moderate precipitaiton/normal temperature
%% 3rd cluster: extreme precipitation/heat events

function [Mu, Pi, GroupProb] = NormalMixture3(X, a0, mu0, kappa0, atau0, btau0, nhat, N, NBurn);

%%-------------------------------------------------------------------------
% Input variables
%   X: input 1-D data (could be precipitation, temperature, etc)
%   a0:  prior of cluster weight
%   mu0: prior of mu
%   Kappa0: 
%   atau0: prior of shape parameter of the Gamma Distribution
%   btau0: prior of scale parameter of the Gamma Distribution
%   nhat: prior of sample size
%   N: number of iterations
%   NBurn: burn-in samples
%
% Output variables:
%   Mu: Cluster mean ~ intensity of precipitation/temperature
%   Pi: Cluster weight ~ frequency of precipitation/temperature
%   GroupProb: Probability of sample falling into a specific group
%%-------------------------------------------------------------------------

k = 3; % 3-cluster Normal mixture
ah = a0 + nhat; 
kappah = kappa0; 
muh    = mu0;
atauh  = atau0;  btauh = btau0;
nr = length(X); 

a = []; Mu = []; kappa = []; a_tau = []; b_tau = []; Pi = [];
DataGroup = zeros(nr,1); ProbNew = zeros(nr,k);
DataGroupIdxTmp = zeros(nr, N); GroupIdx = zeros(nr, 1);
GroupProb = zeros(nr, k);
       
 for iTime = 1:N; %% N-time Gibbs sampler;
        
     pi0=rdirichlet(ah,1); Pi = cat(1,Pi,pi0);   % update the Pi value here
     tau0=gamrnd(atauh,1./btauh);   
     mugroup=normrnd(muh,sqrt(kappah./tau0));
            
     for i = 1:k;
        ProbNew(:,i)=pi0(i)*normpdf(X,mugroup(i),sqrt(1/tau0(i)));
     end;
        
     NormConst = sum(ProbNew,2);
     ProbNew   = ProbNew./repmat(NormConst,1,k);
            
     for ii = 1:nr;   %% assign data groups
        
        Prob1 = ProbNew(ii,1); Prob2 = sum(ProbNew(ii,1:2),2); 
        Prob3 = sum(ProbNew(ii,1:3),2);
        
        latentNum = rand(1);
            
        if (latentNum<=Prob1); DataGroup(ii)=1; end;
        if (latentNum>Prob1&&latentNum<=Prob2); DataGroup(ii)=2; end;
        if (latentNum>Prob2&&latentNum<=Prob3); DataGroup(ii)=3; end;
        
     end;
            
     for i = 1:k;
        nhat(i) = sum(DataGroup==i);
     end;
        
     ah = a0+nhat;
            
     %%% update parameters for full conditional posteriors
     df1 = 1./kappa0+nhat; kappah = 1./df1;
     atauh = atau0+nhat/2;
            
     for i = 1:k;
        
        muh(i)=kappah(i)*(1/kappa0(i)*mu0(i)+sum(X(DataGroup==i)));
        yg = X(DataGroup==i); ybarh = mean(X(DataGroup==i));
            
        if (nhat(i)==0); 
          btauh(i)=btau0(i); 
        else 
          btauh(i)=btau0(i)+0.5*(sum((yg-ybarh).^2)+(nhat(i)/(1+kappa0(i)*nhat(i))*(ybarh-mu0(i)).^2));    
        end; 
        
     end;
     
     %%% dealing with Label-switching by placing restrictiong here
     [x,idx]=sort(muh);
     muh = x; ah = ah(idx); kappah = kappah(idx);
     atauh = atauh(idx); btauh = btauh(idx);
     
     %%% output rainfall groups
     DataGroupIdxTmp(:,iTime) = DataGroup;
      
     %%% output data
     Mu = cat(1,Mu,muh); %% Mu is updated here
     a = cat(1,a,ah); kappa=cat(1,kappa,kappah);
     a_tau = cat(1,a_tau,atauh); b_tau = cat(1,b_tau,btauh);
          
end;

%% assign group Idx to each day

for iday = 1:nr;
    
    RainGroup = DataGroupIdxTmp(iday,NBurn+1:N); %% the first NBurn are burn-in samples
    GroupProb_Raw(1) = sum(RainGroup==1);
    GroupProb_Raw(2) = sum(RainGroup==2);
    GroupProb_Raw(3) = sum(RainGroup==3);
    
    [Group,GroupId] = sort(GroupProb_Raw);
    
    GroupIdx(iday) = GroupId(3);
    GroupProb(iday, :) = GroupProb_Raw./(N-NBurn); 
    
end;

%   mu1Out(:,n) = mu(:,1); mu2Out(:,n) = mu(:,2); mu3Out(:,n) = mu(:,3); 
%   pi1Out(:,n) = piNew(:,1); pi2Out(:,n) = piNew(:,2); pi3Out(:,n) = piNew(:,3);
    
% end;

% MuData = cat(3, mu1Out, mu2Out, mu3Out);
% PiData = cat(3, pi1Out, pi2Out, pi3Out);

%save('PrYangtze_MuMCMC.mat','MuData');
%save('PrYangtze_PiMCMC.mat','PiData');

%tElapsed = toc(tStart)
