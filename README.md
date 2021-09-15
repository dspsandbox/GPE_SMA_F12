# GPE code for the f=1 & f=2 inter-hypefine interaction in <sup>87</sup>Rb

This is a Python library for GPE meanfield simulations of a F=1,2 spinor BEC under the single mode approximation (SMA). The evolution of the spinor wavefunction is obtained by performing the functional derivative of the meanfield energy functional 
* The energy functional is calculated for the f=1 and f=2 hyperfine groundsate manifolds of <sup>87</sup>Rb. Since these manifolds feature oposite gyromagnetic ratios, the energy functional is described in a counter rotating reference frame and under the rotating wave approximation (RWA), which drops the fast oscillating terms.  
* The functional derivative is evaluated analytically using _Scipy_.
* The set of coupled differential equations is integrated by the _Scipy.integrate.odeint_ routine

### Examples
Please have a look at an interactive implementation of the library [nbviewer](https://nbviewer.jupyter.org/github/dspsandbox/GPE_SMA_F12/blob/master/GPE_SMA_F1F2_example.ipynb) or by launching [binder](https://mybinder.org/v2/gh/dspsandbox/GPE_SMA_F12/blob/main/GPE_SMA_F1F2_example.ipynb/HEAD) (launching the interactive server may take up to 1 minutes):







