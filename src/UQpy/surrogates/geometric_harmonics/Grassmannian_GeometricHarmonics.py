import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors #todo: Change to scipy (using KD.Tree)


class GrassmannianGHMap:
    """
    Geometric Harmonics for domain extension.
    The class ``GeometricHarmonics`` is used in the domain extension of functions defined only on few observations.
    ``GeometricHarmonics`` is a Subclass of Similarity.
    **Input:**
    * **n_evecs** (`int`)
        The number of eigenvectors used in the eigendecomposition of the kernel matrix.
    * **kernel_method** (`callable`)
        Kernel method used in the construction of the geometric harmonics.
    **Attributes:**
    * **n_evecs** (`int`)
        The number of eigenvectors used in the eigendecomposition of the kernel matrix.
    * **X** (`list`)
        Independent variables.
    * **y** (`list`)
        Function values.
    * **basis** (`list`)
        Basis used in the domain extension.
    **Methods:**
    """

    def __init__(self,n_eigenvectors=None, n_neighbors=None, p='max',
                 epsilon_gh0=None, epsilon_gh1=None,
                 diffusion_kernel_object=ProjectionKernel()):

        # kernel_object=Gaussian()
        #self.kernel_object = kernel_object
        self.p = p
        self.diffusion_kernel_object = diffusion_kernel_object
        #self.n_evecs = n_evecs
        self.X = None
        self.y = None

        self.diffusion_object = None
        self.grassmann_object = None
        self.gh0 = None
        self.gh1 = None
        self.epsilon_gh0 = epsilon_gh0
        self.epsilon_gh1 = epsilon_gh1
        self.n_eigenvectors = n_eigenvectors
        self.n_neighbors = n_neighbors
        self.dcoords = None

    def fit(self, X, y, **kwargs):

        """
        Train the model using `fit`.
        In this method, `X` is a list of data points, `y` are the function values. `epsilon` can be
        provided, otherwise it is computed from the median of the pairwise distances of X.
        **Input:**
        * **X** (`list`)
            Input data (independent variables).
        * **y** (`list`)
            Function values.
        * **epsilon** (`float`)
            Parameter of the Gaussian kernel.
        **Output/Returns:**
        """

        if X is not None:

            self.X = X
            self.y = y
            #if not isinstance(X, list):
            #    raise TypeError('UQpy: `X` must be a list.')

            #if not isinstance(y, list):
            #    raise TypeError('UQpy: `y` must be a list.')

            self.grassmann_object = Grassmann(p=self.p)
            self.grassmann_object.fit(X=self.y)

            self.diffusion_object = GrassmannianDiffusionMaps(alpha=0.5, n_evecs=len(X), sparse=False, k_neighbors=1,
                                                              kernel_composition='prod',
                                                              kernel_object=self.diffusion_kernel_object,
                                                              p=self.p, orthogonal=False) #todo: change n_evecs!!!!

            self.diffusion_object.fit(X=self.grassmann_object)

            self.dcoords = self.diffusion_object.dcoords

            self.gh0 = GeometricHarmonics(n_evecs=self.n_eigenvectors)
            self.gh0.fit(X, self.dcoords, epsilon=self.epsilon_gh0)

            self.gh1 = GeometricHarmonics(n_evecs=self.n_eigenvectors)
            self.gh1.fit(X, self.grassmann_object.sigma, epsilon=self.epsilon_gh0)

    def predict(self, X):

        dcoords = self.dcoords
        dcoord_interp = self.gh0.predict(X, )
        sig_interp = self.gh1.predict(X, )

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(dcoords)
        distances, indices = nbrs.kneighbors(dcoord_interp)
        pid = indices[0]

        psi_gr = []
        phi_gr = []
        for k in pid:
            psi_gr.append(self.grassmann_object.psi[k])
            phi_gr.append(self.grassmann_object.phi[k])

        # start = timeit.default_timer()
        # ref_psi = Gr.karcher_mean(points_grassmann=psi_gr)
        # ref_phi = Gr.karcher_mean(points_grassmann=phi_gr)
        ref_psi = psi_gr[0]
        ref_phi = phi_gr[0]
        # stop = timeit.default_timer()
        # print('Time: ', stop - start)

        gpsi = self.grassmann_object.log_map(points_grassmann=psi_gr, ref=ref_psi)
        gphi = self.grassmann_object.log_map(points_grassmann=phi_gr, ref=ref_phi)

        psi = []
        sigma = []
        phi = []
        diffc = []

        gamma_psi = []
        gamma_phi = []
        ptv = []
        n0 = np.shape(self.grassmann_object.psi[pid[0]])[0]
        n1 = np.shape(self.grassmann_object.psi[pid[0]])[1]
        n0h = np.shape(self.grassmann_object.phi[pid[0]])[0]
        n1h = np.shape(self.grassmann_object.phi[pid[0]])[1]
        for c, k in enumerate(pid):
            #n0 = np.shape(self.grassmann_object.psi[k])[0]
            #n1 = np.shape(self.grassmann_object.psi[k])[1]

            gamma_psi.append(gpsi[c].reshape(n0 * n1))
            gamma_phi.append(gphi[c].reshape(n0h * n1h))

            psi.append(self.grassmann_object.psi[k].reshape(n0 * n1))
            sigma.append(self.grassmann_object.sigma[k].reshape(len(self.grassmann_object.sigma[k])))
            phi.append(self.grassmann_object.phi[k].reshape(n0h * n1h))
            diffc.append(self.dcoords[k, :])
            ptv.append(self.X[k, :])

        #g0 = dcoord_interp
        #dg = pdist(diffc)
        epsilon_local = (np.median(pdist(diffc)) ** 2) / 4

        gh_local = GeometricHarmonics()

        gh_local.fit(diffc, gamma_psi, epsilon=epsilon_local)
        gpsipred_ = gh_local.predict(dcoord_interp, )

        gh_local.fit(diffc, gamma_phi, epsilon=epsilon_local)
        gphipred_ = gh_local.predict(dcoord_interp, )

        #gh_local.fit(diffc, sigma, epsilon=epsilon_local)
        #sigpred_ = gh_local.predict(dcoord_interp)

        gpsipred = gpsipred_[0].reshape(n0, n1)
        gphipred = gphipred_[0].reshape(n0h, n1h)
        psipred_ = self.grassmann_object.exp_map(points_tangent=[gpsipred], ref=ref_psi)[0]
        phipred_ = self.grassmann_object.exp_map(points_tangent=[gphipred], ref=ref_phi)[0]
        #sigpred_ = np.diag(sigpred_[0])

        #Stest = out[pid[0]]
        ut, st, vt = svd(self.grassmann_object.X[pid[0]], rank=self.grassmann_object.p)

        psi_ = []
        phi_ = []
        for i in range(len(ut[0, :])):
            err1 = np.linalg.norm(psipred_[:, i] - ut[:, i])
            err2 = np.linalg.norm(phipred_[:, i] - vt[:, i])

            if err1 > 1:
                psi_.append(-psipred_[:, i])
            else:
                psi_.append(psipred_[:, i])

            if err2 > 1:
                phi_.append(-phipred_[:, i])
            else:
                phi_.append(phipred_[:, i])
            # phipred_[:,i]

        psi_ = np.array(psi_).T
        phi_ = np.array(phi_).T
        sigma_ = sig_interp[0]

        y = psi_ @ np.diag(sigma_) @ phi_.T

        return y, psi_, sigma_, phi_, dcoord_interp