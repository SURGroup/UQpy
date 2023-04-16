import numpy as np
import UQpy
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import polynomial_chaos
from scipy.spatial.distance import cdist
from beartype import beartype

class ThetaCriterionPCE:
    @beartype
    def __init__(self,surrogates: list):
        """
        Active learning for polynomial chaos expansion using Theta criterion balancing between exploration and exploitation.
        
        :param surrogates: list of objects of the :py:meth:`UQpy` :class:`PolynomialChaosExpansion` class 
        """
        
        self.surrogates=surrogates
        
    
    def run(self, X: np.ndarray, Xcandidate: np.ndarray, nadd=1, WeightsS=None, WeightsSCandidate=None, WeightsPCE=None, Criterium=False):

        """
        Execute the :class:`.ThetaCriterionPCE` active learning.
        :param X: Samples in existing ED used for construction of pces.
        :param Xcandidate: Candidate samples for selectiong by Theta criterion.
        :param WeightsS: Weights associated to X samples (e.g. from Coherence Sampling).
        :param WeightsSCandidate: Weights associated to candidate samples (e.g. from Coherence Sampling).
        :param nadd: Number of samples selected from candidate set in a single run of this algorithm
        :param WeightsPCE: Weights associated to each PCE (e.g. Eigen values from dimension-reduction techniques)
        
        The :meth:`run` method is the function that performs iterations in the :class:`.ThetaCriterionPCE` class.
        The :meth:`run` method of the :class:`.ThetaCriterionPCE` class can be invoked many times for sequential sampling.
        
        :return: Position of the best candidate in candidate set. If ``Criterium = True``, values of Theta criterion (variance density, average variance density, geometrical part, total Theta criterion) for all candidates are returned instead of a position. 

        """

        pces=self.surrogates

        npce=len(pces)
        nsimexisting, nvar = X.shape
        nsimcandidate, nvar = Xcandidate.shape
        l = np.zeros(nsimcandidate)
        criterium = np.zeros(nsimcandidate)
        if WeightsS is None:
            WeightsS = np.ones(nsimexisting)

        if WeightsSCandidate is None:
            WeightsSCandidate = np.ones(nsimcandidate)
            
        if WeightsPCE is None:
            WeightsPCE = np.ones(npce)

        pos=[]
        
        for n in range (nadd):

            
            S=polynomial_chaos.Polynomials.standardize_sample(X,pces[0].polynomial_basis.distributions)   
            Scandidate=polynomial_chaos.Polynomials.standardize_sample(Xcandidate,pces[0].polynomial_basis.distributions)   

            lengths = cdist(Scandidate, S)
            closestS_pos = np.argmin(lengths, axis=1)
            closest_valueX = X[closestS_pos]
            l = np.nanmin(lengths, axis=1)
            variance_candidate=0
            variance_closest=0

            for i in range(npce):
                variance_candidatei=0
                variance_closesti=0
                pce=pces[i]
                variance_candidatei = self.LocalVariance(Xcandidate, pce, WeightsSCandidate) 
                variance_closesti = self.LocalVariance(closest_valueX, pce, WeightsS[closestS_pos]) 

                variance_candidate=variance_candidate+variance_candidatei*WeightsPCE[i]
                variance_closest=variance_closest+variance_closesti*WeightsPCE[i]

            criteriumV = np.sqrt(variance_candidate * variance_closest)
            criteriumL = l**nvar
            criterium = criteriumV * criteriumL
            pos.append(np.argmax(criterium))
            X=np.append(X,Xcandidate[pos,:],axis=0)
            WeightsS=np.append(WeightsS,WeightsSCandidate[pos])
            
        if Criterium == False:
            if nadd==1:
                pos=pos[0]
            return pos
        else:
            return variance_candidate, criteriumV, criteriumL, criterium
    
    
    # calculate variance density of PCE for Theta Criterion
    def LocalVariance(self,coord,pce,Weight=1):
        Beta=pce.coefficients
        Beta[0] = 0


        product=pce.polynomial_basis.evaluate_basis(coord)

        product = np.transpose(np.transpose(product)*Weight)
        product = product.dot(Beta)

        product = np.sum(product,axis=1)

        product= product**2
        product = product *polynomial_chaos.Polynomials.standardize_pdf(coord,pce.polynomial_basis.distributions)

        return product
