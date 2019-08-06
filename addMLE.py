"""This applications runs a MLE for the aDDM (Krajbich et al., 2010
"""

#Packages
import scipy as spy
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
#Ignore dumb warninings
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option('mode.chained_assignment', None)

####################
#Important functions#
#####################
#Midround function that Steph loves
def midround(x,base):
    z = base*round(x/base)
    return 
#Get dwell time adv
def fixsumsFN(data):
    #Input
    ##Data by trial and fixation
    
    data['leftfixtime'] = (data.roi%2)*data.fixdur
    data['rightfixtime'] = ((data.roi+1)%2)*data.fixdur
    #do a quick cumulative sum over those fixation durations over the whole dataset
    data['tleft'] = np.cumsum(data.leftfixtime)
    data['tright'] = np.cumsum(data.rightfixtime)
    
    #now only look at the last fixation of every trial
    temp = data[data.revfixnum==1]
    
    #subtract the sum at the end of previous trial from the sum at the end of the current trial - that gives you just the sum from the current trial
    tempIndex = np.array(list(range(1,temp.shape[0]-1)))
    templ = np.array(temp['tleft'])
    tempr = np.array(temp['tright'])
    templ = np.insert(templ,0,0)
    tempr = np.insert(tempr,0,0)
    templ = np.delete(templ,len(templ)-1)
    tempr = np.delete(tempr,len(tempr)-1)
    temp['tleft'] = temp['tleft']-templ
    temp['tright'] = temp['tright']-tempr
    
    #calculate the final total left time advantage variable
    temp['leftadv'] = temp.tleft-temp.tright
    
    fixsums = temp 
    return fixsums

#Simulate addm
def addmSim_v1(nsim,d,sigma,theta,ndt,dsigma,data_mat):
    #Input:
    ## nsim (number of simulations per trial)
    ## d (drift rate slope)
    ## sigma (within-trial noise)
    ## theta (attentional bias)
    ## ndt (non-decision time)
    ## dsigma (across-trial noise)
    ## data_mat (observed fixations per trial)
    import zignor
    import math
    ndt = int(ndt)
    boundary = 1 #set boundary to 1 by default
    simCount = [x+1 for x in list(range(nsim))] 
    
    ntrial = int(data_mat.shape[0]) #number of unique trials
    nCol = int(data_mat.shape[1])
    maxfix = (int(data_mat.shape[1])-4)/2 #max number of fixations per trial
    
    #Pre-allocate the simulation vectors
    rt = np.repeat(0,nsim*maxfix*ntrial) #reaction time vector
    choice = np.repeat(-1,nsim*maxfix*ntrial) #choice vector
    fixnum = np.repeat(-1,nsim*maxfix*ntrial) #fixation number vector
    revfixnum = np.repeat(-1,nsim*maxfix*ntrial) #reverse fixation number vector
    fixdur = np.repeat(-1,nsim*maxfix*ntrial) #fixation duration vector
    nfix = np.repeat(-1,nsim*maxfix*ntrial) #total number of fixations vector
    roi = np.repeat(-1,nsim*maxfix*ntrial) #roi vector
    leftval = np.repeat(-1,nsim*maxfix*ntrial) #left value vector
    rightval = np.repeat(-1,nsim*maxfix*ntrial) #right value vector
    
    #matrix of all ROIs by trial
    roiIndex = [x + 4 for x in range(int(maxfix))]
    roi_mat = data_mat[:,roiIndex]
    #matrix of all fix lengths trial
    fixlength_mat = data_mat[:,(4+int(maxfix)):int(data_mat.shape[1]+1)]
    #matrix of values
    value_mat = np.ones((ntrial,int(roi_mat.shape[1])))
    for col in range(int(roi_mat.shape[1])):
        for row in range(ntrial):
            if roi_mat[row,col] == 1:
                value_mat[row,col] = data_mat[row,2]
            if roi_mat[row,col] == 2:
                value_mat[row,col] = data_mat[row,3]
    lvalue_mat = np.ones((ntrial,int(roi_mat.shape[1])))
    rvalue_mat = np.ones((ntrial,int(roi_mat.shape[1])))
    for row in range(ntrial):
        lvalue_mat[row,:] = data_mat[row,2]
        rvalue_mat[row,:] = data_mat[row,3]
    #matrix of drift rates
    drift_mat = np.ones((ntrial,int(roi_mat.shape[1])))
    for row in range(ntrial):
        for col in range(int(roi_mat.shape[1])):
            if roi_mat[row,col] == 1:
                drift_mat[row,col] = d*(lvalue_mat[row,col]-theta*rvalue_mat[row,col])
            if roi_mat[row,col] == 2:
                drift_mat[row,col] = d*(theta*lvalue_mat[row,col]-rvalue_mat[row,col])
    
    #loop for each trial
    choice = []
    rt = []
    fixnum = []
    nfix = []
    revfixnum =[]
    roi = []
    leftval = []
    rightval = []
    fixdur = []
    colNames = ['choice','rt','fixnum','nfix',
            'revfixnum','roi','leftval',
            'rightval','fixdur']
    sims = pd.DataFrame(columns=colNames)

    for trial in range(ntrial):
        nroi = roi_mat[trial,:]
        nfixlength = fixlength_mat[trial,:]
        ndrift = drift_mat[trial,:]
        ntemp = np.cumsum(nfixlength)
        index = np.repeat(0,max(ntemp))
        for x in ntemp[range(len(ntemp)-1)]:
            index[int(x)] = 1
        index[0] = 1
        index = np.cumsum(index)
        bigdrift = np.float64(index)
        for x in np.unique(index):
            bigdrift[bigdrift==x] = ndrift[x-1]
            
        #presample vector of noise
        maxt = int(max(ntemp))
        noises = zignor.rnor(maxt*nsim)*sigma
        
        #presample across trial noise
        dnoises = zignor.rnor(nsim)*dsigma
        
        j = 0
        
        #loop over number of simulations per trial  
        for sim in range(nsim):
            j = int(j+1)
            noise = noises[((j-1)*maxt):((j)*maxt)]
            evidence = bigdrift+noise+dnoises[sim]
            RDV = np.cumsum(evidence)
            absRDV = abs(RDV)
            for u in absRDV:
                if u >= boundary:
                    trt = int(np.argwhere(absRDV==u))
                    break
            else:
                trt = maxt-1
            
            #save the trial data
            fixNum = int(min(np.argwhere(np.cumsum(nfixlength)>=trt)))
            if fixNum < 1: #catch errors where trial fails
                fixNum = 1
            nfix = np.append(nfix, [fixNum]*fixNum)
            choice = np.append(choice, [math.ceil(RDV[trt]/1000000)]*fixNum)
            rt = np.append(rt, [trt+ndt]*fixNum)
            fixnum = np.append(fixnum,list(range(1,fixNum+1)))
            revfixnum = np.append(revfixnum, list(range(fixNum,0,-1)))
            roi = np.append(roi,nroi[list(range(1,fixNum+1))])
            leftval = np.append(leftval,[lvalue_mat[trial,1]]*fixNum)                   
            rightval = np.append(rightval,[rvalue_mat[trial,1]]*fixNum)   
            #fix the last fixation time
            tfixLength = nfixlength[list(range(fixNum))]
            erro = sum(tfixLength)-trt
            tfixLength[fixNum-1] = tfixLength[fixNum-1]-erro
            fixdur = np.append(fixdur,tfixLength)
            
            
    data = {'choice': choice,
            'rt':rt,
            'fixnum':fixnum,
            'nfix':nfix,
            'revfixnum':revfixnum,
            'roi':roi,
            'leftval':leftval,
            'rightval':rightval,
            'fixdur':fixdur}
    sims = pd.DataFrame(data)
    sims = sims.loc[sims.choice>=0]
    sims = sims.loc[sims['rt']<(maxt+ndt)] #remove trials that did not finish
            
    del(noises,bigdrift,choice,data_mat,drift_mat,fixdur,fixlength_mat,fixnum,index,leftval,lvalue_mat,maxfix,maxt,ndrift,nfix,nfixlength,nroi,nsim,ntemp,ntrial,revfixnum,rightval,roi,roi_mat,rt,rvalue_mat,value_mat)

    return(sims)

#Full fit function
#Fit function
def addmFit_v1(data,sims):
    #Input:
    ## Empirical Data
    ## Simulated Data
    from scipy.stats import chisquare 
    
    ####chi-square and MLE fit with correct vs. incorrect by valdiff and tadvsep
    pquants = [0,0.5,1] #desired correct and zero value-difference quantiles (INCLUDE 0 and 1)
    nquants = [0,0.5,1] #desired incorrect quantiles (INCLUDE 0 and 1)
    tadvseps = [0.333,0.667] #desired separations (DON'T INCLUDE 0 or 1)
    unbound = 0 #set to 1 if you want the lower bound to be zero and the upper bound to be 100secs.
    maxvaldiff = 100000 #use this to set a maximum value difference 
    fourmode = 3 #use this to set how you want to bin trials with 4 items. #1: only use the top items, #2:only use the bottom items, #3: use the average of the top and bottom

    ntadvseps = len(tadvseps)+1 # number of time advantage "chunks" to slice the data into for binning
    CS = 0 # this is where we store the chi-square statistic
    ML = 0 # this is where we store the MLE statistic
    sims = fixsumsFN(sims)
    data = fixsumsFN(data)
    gof = []
    #2 item case
    if max(data.roi)<3:
        data['valdiff'] = data.leftval - data.rightval
        data['absvaldiff'] = midround(abs(data.valdiff),0.001)
        sims['valdiff'] = sims.leftval - sims.rightval
        sims['absvaldiff'] = midround(abs(sims.valdiff),0.001)
        
    #Remove extreme value differences
    data = data[data.absvaldiff <= maxvaldiff]
    sims = sims[sims.absvaldiff <= maxvaldiff]
    
    #This bit helps with missing valdiff 0 sim trials
    absvaldiffs = np.intersect1d(np.unique(sims.absvaldiff),np.unique(data.absvaldiff))
    
    if np.median(data.tleft) > 50: #if median dwell time is over 50, RTs are in ms
        data['tadvl'] = data.tleft - data.tright
    else: #otherwise, we convert to ms
        data['tadvl'] = (data.tleft - data.tright)*1000
        
    data['tadvc'] = (data.valdiff>=0)*data.tadvl + (data.valdiff < 0)*(-1*data.tadvl)
    sims['tadvl'] = sims.tleft - sims.tright
    sims['tadvc'] = (sims.valdiff>=0)*sims.tadvl + (sims.valdiff < 0)*(-1*sims.tadvl)
    
    #All left choices in 0-valuedifference trials are coded as correct, all right choices
    #are coded as incorrect
    
    #preallocate matrices to store the quantiles and the counts/probabilities
    dataprobs = np.zeros((len(absvaldiffs[absvaldiffs>0]),ntadvseps*(len(pquants)+len(nquants)-2)))
    posquants = np.zeros((len(absvaldiffs[absvaldiffs>0]),ntadvseps*len(pquants)))
    negquants = np.zeros((len(absvaldiffs[absvaldiffs>0]),ntadvseps*len(nquants)))
    
    
    #Check if there are zero value difference trials and if so, skip them in loop
    start = 0
    if (absvaldiffs[0] == 0):
        start = 1
    
    postemp = []
    negtemp = []
    if len(absvaldiffs)>1:
        for i in range(start, len(absvaldiffs)):
            postemp2 = []
            negtemp2 = []
            temp = data.loc[data.absvaldiff == absvaldiffs[i]]
            posdat = temp.loc[((temp['valdiff']>0) & (temp['choice']==1)) | ((temp['valdiff']<0) & (temp['choice']==0))]
            negdat = temp.loc[((temp['valdiff']>0) & (temp['choice']==0)) | ((temp['valdiff']<0) & (temp['choice']==1))]
            # catch if there are only correct/errors for a given difficulty
            if posdat.shape[0] == 0:
                posdat.loc[0] = [0]*posdat.shape[1]
            if negdat.shape[0] == 0:
                negdat.loc[0] = [0]*negdat.shape[1]
            posseps = np.quantile(posdat.tadvc,tadvseps)
            posseps = np.insert(posseps,0,-1000000)
            posseps = np.insert(posseps,(len(posseps)),1000000)
            negseps = np.quantile(negdat.tadvc,tadvseps)
            negseps = np.insert(negseps,0,-1000000)
            negseps = np.insert(negseps,(len(negseps)),1000000)
            for sepnum in range(ntadvseps):
                posdattemptadv = posdat.loc[(posdat['tadvc'] >= posseps[sepnum]) & (posdat['tadvc'] < posseps[sepnum+1])]
                negdattemptadv = negdat.loc[(negdat['tadvc'] >= negseps[sepnum]) & (negdat['tadvc'] < negseps[sepnum+1])]
                pquantIndex = list(range(((sepnum)*len(pquants)),((sepnum+1)*len(pquants))))
                posquants[(i-start),pquantIndex] = np.quantile(posdattemptadv['rt'],pquants)
                nquantIndex = list(range(((sepnum)*len(nquants)),((sepnum+1)*len(nquants))))
                negquants[(i-start),nquantIndex] = np.quantile(negdattemptadv['rt'],nquants)
      
            posquants[pd.isna(posquants)==True] = 0
            negquants[pd.isna(negquants)==True] = 0

            #set lower and upper bounds to 0 and 100 if "unbound" is turned on
            if unbound == 1:
                for sepnum in range(ntadvseps):
                    quantIndex = list(range(((sepnum)*len(pquants)),((sepnum+1)*len(pquants))))
                    posquants[(i-start),((sepnum)*len(pquants))] = 0
                    posquants[(i-start),((sepnum+1)*len(pquants))] = 10000000 #some really big number
                    negquants[(i-start),((sepnum)*len(nquants))] = 0
                    negquants[(i-start),((sepnum+1)*len(nquants))] = 10000000 #some really big number
                    
            for sepnum in range(ntadvseps):
                posdattemptadv = posdat[(posdat['tadvc'] >= posseps[sepnum]) & (posdat['tadvc'] < posseps[sepnum+1])]
                negdattemptadv = negdat[(negdat['tadvc'] >= negseps[sepnum]) & (negdat['tadvc'] < negseps[sepnum+1])]
                pquantIndex = list(range(((sepnum)*len(pquants)),((sepnum+1)*len(pquants))))
                nquantIndex = list(range(((sepnum)*len(nquants)),((sepnum+1)*len(nquants))))
                posdatacount, pBinEdges = np.histogram(posdattemptadv.rt,bins=posquants[(i-start),pquantIndex], density = False)
                negdatacount, nBinEdges = np.histogram(negdattemptadv.rt,bins=negquants[(i-start),nquantIndex], density = False)
                #posdatatotal = sum(posdatacount)
                #negdatatotal = sum(negdatacount)
                #posdataprobs = [x/posdatatotal for x in posdatacount]
                #negdataprobs = [x/negdatatotal for x in negdatacount]
                posdataprobs = posdatacount
                negdataprobs = negdatacount
                quantIndex2 = list(range((sepnum)*(len(posdataprobs)+len(negdataprobs)),(sepnum+1)*(len(posdataprobs)+len(negdataprobs))))
                tempShape = np.reshape([posdataprobs,negdataprobs], dataprobs[(i-start),quantIndex2].shape).T
                dataprobs[(i-start),quantIndex2] = tempShape
                postemp2 = np.append(postemp2, posdataprobs)
                negtemp2 = np.append(negtemp2, negdataprobs)
                
            if i == start:
                postemp = postemp2
                negtemp = negtemp2
            else:
                postemp = np.vstack([postemp, postemp2])
                negtemp = np.vstack([negtemp, negtemp2])
         
        #Where there are no value difference == 0 cases
            if start == 1:
                temp = data.loc[data['absvaldiff']==0]
                zeroseps = np.quantile(temp.tadvc,tadvseps)
                zeroseps = np.insert(zeroseps,0,-1000000)
                zeroseps = np.insert(zeroseps,len(zeroseps),1000000)
                zeroquants = np.zeros((1,ntadvseps*len(pquants)))
                for sepnum in range(ntadvseps):
                    temptadv = temp.loc[(temp['tadvc'] >= zeroseps[sepnum]) & (temp['tadvc'] < zeroseps[sepnum+1])]
                    tempIndex = list(range(sepnum*len(pquants),(sepnum+1)*len(pquants)))
                    zeroquants[0,tempIndex] = np.quantile(temptadv.rt,pquants)
                zeroquants[pd.isna(zeroquants)==True] = 0
                if unbound == 1:
                    for sepnum in range(ntadvseps):
                        zeroquants[(sepnum)*len(pquants)] = 0
                        zeroquants[sepnum*len(pquants)] = 100000 #some really big number
                zerodataprobs = np.zeros((1,ntadvseps*len(pquants)-1))
                for sepnum in range(ntadvseps):
                    temptadv = temp.loc[(temp['tadvc'] >= zeroseps[sepnum]) & (temp['tadvc'] < zeroseps[sepnum+1])]
                    zeroIndex = list(range(((sepnum)*len(pquants)),((sepnum+1)*len(pquants))))
                    zerodatahist, zeroBinEdges = np.histogram(temptadv['rt'],bins=zeroIndex, density = False)
                    zerodatT = sum(zerodatahist)
                    zerodatap = [x/zerodatT for x in zerodatahist]
                    zeroIndex2 = list(range(((sepnum)*len(zerodatap)),((sepnum+1)*len(zerodatap))))
                    zerodataprobs[0,zeroIndex2] = zerodatap
        
        #Now handle the simulations
        simprobs = np.zeros((len(absvaldiffs[absvaldiffs>0]),ntadvseps*(len(pquants)+len(nquants)-2)))
        for i in range(start,len(absvaldiffs)):
            #use the seps from the data, not from sims
            tempdat = data.loc[data['absvaldiff']==absvaldiffs[i]]
            posdat = tempdat.loc[((tempdat['valdiff']>0) & (tempdat['choice']==1))|((tempdat['valdiff']<0) & (tempdat['choice']==0))]
            negdat = tempdat.loc[((tempdat['valdiff']>0) & (tempdat['choice']==0))|((tempdat['valdiff']<0) & (tempdat['choice']==1))]
            # catch if there are only correct/errors for a given difficulty
            if posdat.shape[0] == 0:     
                posdat.loc[0] = [0]*posdat.shape[1]
            if negdat.shape[0] == 0:
                negdat.loc[0] = [0]*negdat.shape[1]
            posseps = np.quantile(posdat.tadvc,tadvseps)
            posseps = np.insert(posseps,0,-1000000)
            posseps = np.insert(posseps,len(posseps),1000000)
            negseps = np.quantile(negdat.tadvc,tadvseps)
            negseps = np.insert(negseps,0,-1000000)
            negseps = np.insert(negseps,len(negseps),1000000)
            #now do it with data
            tempsim = sims.loc[sims['absvaldiff']==absvaldiffs[i]]
            possim = tempsim.loc[((tempsim['valdiff']>0) & (tempsim['choice']==1))|((tempsim['valdiff']<0) & (tempsim['choice']==0))]
            negsim = tempsim.loc[((tempsim['valdiff']>0) & (tempsim['choice']==0))|((tempsim['valdiff']<0) & (tempsim['choice']==1))]
            for sepnum in range(ntadvseps):
                possimtadv = possim.loc[(possim['tadvc'] >= posseps[sepnum]) & (possim['tadvc'] < posseps[sepnum+1])]
                negsimtadv = negsim.loc[(negsim['tadvc'] >= negseps[sepnum]) & (negsim['tadvc'] < negseps[sepnum+1])]
                pIndex = list(range((sepnum)*len(pquants),(sepnum+1)*len(pquants)))
                nIndex = list(range((sepnum)*len(nquants),(sepnum+1)*len(nquants)))
                posrtcounts, posrthist = np.histogram(possimtadv.rt,bins=posquants[(i-start),pIndex])
                #posrttotal = sum(posrtcounts)
                #posrtprobs = [rt/posrttotal for rt in posrtcounts]
                posrtprobs = posrtcounts
                negrtcounts, negrthist = np.histogram(negsimtadv.rt,bins=negquants[(i-start),nIndex])
                #negrttotal = sum(negrtcounts)
                #negrtprobs = [rt/negrttotal for rt in negrtcounts]
                negrtprobs = negrtcounts
                pnIndex = list(range((sepnum)*(len(posrtprobs)+len(negrtprobs)),(sepnum+1)*(len(posrtprobs)+len(negrtprobs))))
                tempShapePN = np.reshape([posrtprobs,negrtprobs], simprobs[(i-start),pnIndex].shape).T
                simprobs[(i-start),pnIndex] = tempShapePN

            #Make sure at least 1 in each bin
            if sum(simprobs[(i-start),:]) == 0: #cheat to make sure the code doesn't crash
                sumSimProbs = 0.0000000000000000001
            else:
                sumSimProbs = (sum(simprobs[(i-start),:]))
            if (any(simprobs[(i-start),:]>0)) & (sum(postemp[(i-start),:])>(ntadvseps*(len(pquants)-1))) & (sum(negtemp[(i-start),:])>(ntadvseps*(len(nquants)-1))):    
                tempCS, tempPval  = chisquare(dataprobs[(i-start),:], simprobs[(i-start),:])
                tempML = sum(dataprobs[(i-start),:]*np.log(simprobs[(i-start),:]/sumSimProbs))
                if (pd.isna(tempML)==True):
                    tempML <- 0
                ML = ML+tempML
                if (pd.isna(tempCS)==True):
                    tempCS <- 0
                CS = CS+tempCS
            else:
                tempCS = 0
                tempPval = 0
                tempML = 0
                ML = ML+tempML
                CS = CS+tempCS

        if start == 1:
            tempdat = data.loc[data['absvaldiff']==0]
            zeroseps = np.quantile(tempdata.tadvc,tadvseps)
            zeroseps = np.insert(zeroseps,0,-1000000)
            zeroseps = np.insert(zeroseps,len(zeroseps),1000000)
            zerosimsprobs = np.zeros(1,ntadvseps*(len(pquants)-1))
            temp = sims.loc[sims['absvaldiff']==0]
            for sepnum in range(ntadvseps): 
                temptadv = temp.loc[(temp['tadvc'] >= zeroseps[sepnum]) & (temp['tadvc'] < zeroseps[sepnum+1])]
                zIndex = list(range(((sepnum)*len(pquants)),(sepnum+1)*len(pquants)))                  
                zsimc,zerorthist = np.histogram(temptadv.rt,bins=zeroquants[zIndex])
                #zsimt = sum(zsimc)
                #zsimp = [rt/zsimt for rt in zsimc]
                zsimp = zsimc
                zIndex2 = list(range((sepnum)*(len(zsimp)),sepnum*(len(zsimp))))                 
                zerosimsprobs[zIndex2] = zsimp
            
            # Make sure at least one in each bin!
            if sum(simprobs[(i-start),:]) == 0: #cheat to make sure the code doesn't crash
                sumZeroProbs = 0.0000000000000000001
            else:
                sumZeroProbs = (sum(zerosimsprobs))
            if (any(zerosimprobs > 0)) & (sum(zerodataprabs)>(ntadvseps*(len(pquants)-1))):
                tempCS, tempPval  = chisquare(zerodataprobs, zerosimsprobs)
                tempML = sum(zerodataprobs*np.log(zerosimsprobs/sumZeroProbs))
            else:
                tempCS = 0
                tempPval = 0
                tempML = 0
            if (pd.isna(tempCS)==True):
                tempCS <- 0
            if (pd.isna(tempML)==True):
                tempML <- 0
            CS = CS+tempCS
            ML = ML+tempML
        else:
            CS = CS+0
            ML = ML+0
    gof = [CS,ML]
    return(gof)
 
 #Combo function (gets CS and LL)  
 def addmSimFit_v1(params,nsim=10,data_mat=data_mat,data=data):
    #Input: parameters for addm
    ## d (drift rate slope)
    ## sigma (within-trial noise)
    ## theta (attentional bias)
    ## ndt (non-decision time)
    ## dsigma (across-trial noise)
    ## nsim (number of simulations per trial)
    
    #Output: loglikelihood
    sims = addmSim_v1(nsim,params[0],params[1],params[2],params[3],params[4],data_mat)
    
    gof = addmFit_v1(data,sims)
    
    CS = gof[0] #chi-square
    LL = gof[1] #log-likelihood
    
    results = np.append(params,[CS,LL])
    
    return gof

#Combo function (only LL)    
def addmSimFit_v1_LL(params,nsim=10,data_mat=data_mat,data=data):
    #Input: parameters for addm
    ## d (drift rate slope)
    ## sigma (within-trial noise)
    ## theta (attentional bias)
    ## ndt (non-decision time)
    ## dsigma (across-trial noise)
    ## nsim (number of simulations per trial)
    
    #Output: loglikelihood
    sims = addmSim_v1(nsim,params[0],params[1],params[2],params[3],params[4],data_mat)
    
    gof = addmFit_v1(data,sims)
    
    CS = gof[0] #chi-square
    LL = gof[1] #log-likelihood
    
    results = np.append(params,[CS,LL])
    
    return LL
 
 def estimateParamsNM(params,nsim=10,func = addmSimFit_v1_LL,data_mat=data_mat,data=data,sims=sims):
    from scipy import optimize
    
    res = optimize.minimize(fun = func, x0 = [params], method='Nelder-Mead', tol=1e-6)
    
    return res
#########################    
#Define paramer settings#
#########################
nsim = 10 #number of simulations per trial
nparams = 1000 #number of parameter combinations to try
# Initial starting parameters
#define lower and upper bounds on parameters
dmin = 0.00001 # 0.0001
dmax = 0.001 # 0.0004
sigmin = 0.01  # 0.01
sigmax = 0.05 # 0.025
boundmin = 1
boundmax = 1
thetamin = .01 # 0
thetamax = .9 # 1
ndtmin = 0 # 300
ndtmax = 500 # 700
dsigmin = 0 # 0.0001
dsigmax = 0 # 0.001
#Define nsimulations
paramset = np.zeros((nparams,7))
if boundmin == boundmax:
    paramset[:,0] = np.random.uniform(low = dmin, high = dmax, size = nparams) #d
    paramset[:,1] = np.random.uniform(low = sigmin, high = sigmax, size = nparams) #sigma
    paramset[:,2] = np.random.uniform(low = thetamin, high = thetamax, size = nparams) #theta
    paramset[:,3] = np.random.uniform(low = ndtmin, high = ndtmax, size = nparams) #ndt
    paramset[:,4] = np.random.uniform(low = dsigmin, high = dsigmax, size = nparams) #dsig
elif sigmin == sigmax:
    paramset[:,0] = np.random.uniform(low = dmin, high = dmax, size = nparams) #d
    paramset[:,1] = np.random.uniform(low = boundmin, high = boundmax, size = nparams) #sigma
    paramset[:,2] = np.random.uniform(low = thetamin, high = thetamax, size = nparams) #theta
    paramset[:,3] = np.random.uniform(low = ndtmin, high = ndtmax, size = nparams) #ndt
    paramset[:,4] = np.random.uniform(low = dsigmin, high = dsigmax, size = nparams) #dsig
    
    
#parallelize this process of finding best starting point for MLE
import concurrent.futures
import time, random

def procedure(j):
    parameter = paramset[j,:]
    res = addmSimFit_v1(parameter[0:5])
    parameter[5] = res[0]
    parameter[6] = res[1]
    return parameter

def main(parameter):
    output = []
    start = time.time()
    #Can change ProcessPoolExecutor() for ThreadPoolExecutor() but it is mad slow
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for out in executor.map(procedure, range(len(paramset))):
        #for out in executor.map(procedure, range(5)): #for testing purposes
            output.append(out)
    output = np.vstack(output)
    finish = time.time()
    print(f'time ellapsed: {(finish-start)}')
    return output

if __name__ == '__main__':
    paramset = paramset
    output = main(paramset)
    outNoInf = output[(np.isinf(output[:,6])==False)]
    gofcut = np.quantile(outNoInf[:,6],.99)
    topParams = outNoInf[outNoInf[:,6]>=gofcut]
    meanTopParams = np.mean(topParams,axis=0)
    print("\nMean Best Fitting Parameters: ", meanTopParams)
    
#Test Nelder-Mead
start = time.time()
result = estimateParamsNM(meanTopParams[0:5])
finish = time.time()
print(f'time ellapsed: {(finish-start)}')    
