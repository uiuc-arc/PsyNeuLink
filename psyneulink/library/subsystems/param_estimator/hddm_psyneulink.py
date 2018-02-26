import numpy as np
import pymc as pm
import warnings
from psyneulink.library.subsystems.param_estimator.system_likelihood import SystemLikelihoodEstimator
from hddm.models import HDDM, AccumulatorModel
from hddm import wfpt, likelihoods
from kabuki.utils import stochastic_from_dist


class HDDMPsyNeuLink(HDDM):
    """Create hierarchical drift-diffusion model whose likelihood function is computed
    from an underlying PsyNeuLink system.
    :Arguments:
        data : pandas.DataFrame
            Input data with a row for each trial.
            Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
              * 'subj_idx': A unique ID (int) of each subject.
              * Other user-defined columns that can be used in depends_on
                keyword.
    :Optional:
        informative : bool <default=True>
            Whether to use informative priors (True) or vague priors
            (False).  Information about the priors can be found in the
            methods section.  If you run a classical DDM experiment
            you should use this. However, if you apply the DDM to a
            novel domain like saccade data where RTs are much lower,
            or RTs of rats, you should probably set this to False.
        is_group_model : bool
            If True, this results in a hierarchical
            model with separate parameter distributions for each
            subject. The subject parameter distributions are
            themselves distributed according to a group parameter
            distribution.
        depends_on : dict
            Specifies which parameter depends on data
            of a column in data. For each unique element in that
            column, a separate set of parameter distributions will be
            created and applied. Multiple columns can be specified in
            a sequential container (e.g. list)
            :Example:
                >>> hddm.HDDM(data, depends_on={'v': 'difficulty'})
                Separate drift-rate parameters will be estimated
                for each difficulty. Requires 'data' to have a
                column difficulty.
        bias : bool
            Whether to allow a bias to be estimated. This
            is normally used when the responses represent
            left/right and subjects could develop a bias towards
            responding right. This is normally never done,
            however, when the 'response' column codes
            correct/error.
        p_outlier : double (default=0)
            The probability of outliers in the data. if p_outlier is passed in the
            'include' argument, then it is estimated from the data and the value passed
            using the p_outlier argument is ignored.
        default_intervars : dict (default = {'sz': 0, 'st': 0, 'sv': 0})
            Fix intertrial variabilities to a certain value. Note that this will only
            have effect for variables not estimated from the data.
        plot_var : bool
             Plot group variability parameters when calling pymc.Matplot.plot()
             (i.e. variance of Normal distribution.)
        trace_subjs : bool
             Save trace for subjs (needed for many
             statistics so probably a good idea.)
        std_depends : bool (default=False)
             Should the depends_on keyword affect the group std node.
             If True it means that both, group mean and std will be split
             by condition.
        wiener_params : dict
             Parameters for wfpt evaluation and
             numerical integration.
         :Parameters:
             * err: Error bound for wfpt (default 1e-4)
             * n_st: Maximum depth for numerical integration for st (default 2)
             * n_sz: Maximum depth for numerical integration for Z (default 2)
             * use_adaptive: Whether to use adaptive numerical integration (default True)
             * simps_err: Error bound for Simpson integration (default 1e-3)
    :Example:
        >>> data, params = hddm.generate.gen_rand_data() # gen data
        >>> model = hddm.HDDM(data) # create object
        >>> mcmc.sample(5000, burn=20) # Sample from posterior
    """

    def __init__(self, data, system, bias=False, include=(),
                 wiener_params=None, p_outlier=0., **kwargs):

        # Make a note that the class is initializing. We will insert a check in the likelhihood function of the system
        # that checks to see whether the class is still initializing and this means we
        self.is_initializing = True

        # HACK!!!: To implement extensions to HDDM for PsyNeuLink models that compute the WFPT likelihood I decided
        # to extend the library without modifiying its code. To do this we create a new HDDM class that inherits from
        # the unmodified HDDM class. In order to insert our likelihood into the execution path I needed to modifiy how
        # the constructors for HDDM and HDDMBase work. Rather than change the libraries I reimplemented with one minor
        # change, the insertion of our PsyNeuLink likelihood function. The below code implements the functionality of
        # these two constructors.

        # HDDM class setup. This is code copied from the HDDM constructor.
        self.slice_widths = {'a': 1, 't': 0.01, 'a_std': 1, 't_std': 0.15, 'sz': 1.1, 'v': 1.5,
                             'st': 0.1, 'sv': 3, 'z_trans': 0.2, 'z': 0.1,
                             'p_outlier': 1., 'v_std': 1}
        self.emcee_dispersions = {'a': 1, 't': 0.1, 'a_std': 1, 't_std': 0.15, 'sz': 1.1, 'v': 1.5,
                                  'st': 0.1, 'sv': 3, 'z_trans': 0.2, 'z': 0.1,
                                  'p_outlier': 1., 'v_std': 1}

        self.is_informative = kwargs.pop('informative', True)

        # HDDMBase class setup. This is code copied from the HDDMBase constructor.
        self.default_intervars = kwargs.pop('default_intervars', {'sz': 0, 'st': 0, 'sv': 0})

        self._kwargs = kwargs

        self.include = set(['v', 'a', 't'])
        if include is not None:
            if include == 'all':
                [self.include.add(param) for param in ('z', 'st','sv','sz', 'p_outlier')]
            elif isinstance(include, str):
                self.include.add(include)
            else:
                [self.include.add(param) for param in include]

        if bias:
            self.include.add('z')

        possible_parameters = ('v', 'a', 't', 'z', 'st', 'sz', 'sv', 'p_outlier')
        assert self.include.issubset(possible_parameters), """Received and invalid parameter using the 'include' keyword.
        parameters received: %s
        parameters allowed: %s """ % (tuple(self.include), possible_parameters)

        #set wiener params
        if wiener_params is None:
            self.wiener_params = {'err': 1e-4, 'n_st':2, 'n_sz':2,
                                  'use_adaptive':1,
                                  'simps_err':1e-3,
                                  'w_outlier': 0.1}
        else:
            self.wiener_params = wiener_params
        wp = self.wiener_params
        self.p_outlier = p_outlier

        #set cdf_range
        cdf_bound = max(np.abs(data['rt'])) + 1;
        self.cdf_range = (-cdf_bound, cdf_bound)

        # Now, setup the WFPT class for the HDDM object in such a way that it calls back into PsyNeuLink for computing
        # the WFPT likelihood.

        # We need to take the system that is passed to us and create a likelihood estimator from it. This will handle
        # the creation of a function that allows us to callback into PsyNeuLink and run the system to estimate the
        # likelihood.
        self.system = system
        self.pnl_likelihood_estimator = SystemLikelihoodEstimator(self.system)

        # Generate the WFPT class the same way HDDM does it.
        hddm_wfpt_class = likelihoods.generate_wfpt_stochastic_class(wp, cdf_range=self.cdf_range)

        # Generate our own using the PsyNeuLink likelihood
        wfpt_class = stochastic_from_dist('wfpt', self.pnl_likelihood_estimator.get_likelihood_function(**wp))

        # Do some surgery on the object where we copy the other auxillary HDDM functions needed to make it a fully
        # functioning stochastic object. This will probably need to be changed in the future to have PsyNeuLink
        # completely implement this interface instead of just the likilihood.
        wfpt_class.pdf = hddm_wfpt_class.pdf
        wfpt_class.cdf_vec = hddm_wfpt_class.cdf_vec
        wfpt_class.cdf = hddm_wfpt_class.cdf
        wfpt.random = hddm_wfpt_class.random
        likelihoods.add_quantiles_functions_to_pymc_class(wfpt_class)

        # Assign it to the object.
        self.wfpt_class = wfpt_class

        # We don't want to call the HDDMBase of HDDM constructor because they create the WFPT class which
        # specifies the likelihood. This would in effect overwrite the PsyNeuLink computed likelihood that we just
        # setup above. Instead, lets go two levels down and call the AccumulatorModel constructor that they inherit from
        # to setup the rest of the model based on our likelihood.
        AccumulatorModel.__init__(self, data, **kwargs)

    def sample(self, *args, **kwargs):
        """Sample from posterior.
        :Note:
            Forwards arguments to pymc.MCMC.sample().
        """

        # Fetch out arguments for db backend
        db = kwargs.pop('db', 'ram')
        dbname = kwargs.pop('dbname', None)

        # init mc if needed
        if self.mc == None:
            self.mcmc(db=db, dbname=dbname)

        # suppress annoying warnings
        if ('hdf5' in dir(pm.database)) and \
                isinstance(self.mc.db, pm.database.hdf5.Database):
            warnings.simplefilter('ignore', pm.database.hdf5.tables.NaturalNameWarning)

        # sample
        self.mc.sample(*args, **kwargs)

        self.sampled = True

        return self.mc

