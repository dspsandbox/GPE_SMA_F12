# GPE code for the f=1 & f=2 inter-hypefine interaction in <sup>87</sup>Rb

This is a Python Library for GPE meanfield simulations of a F=1,2 spinor BEC under the single mode approximation (SMA). The evolution of the spinor wavefunction is obtained by performing the functional derivative of the meanfield energy functional. 
* The functional derivative is evaluated analytically using _Scipy_.
* The set of coupled differential equations is integrated by the _Scipy.integrate.odeint_ routine


