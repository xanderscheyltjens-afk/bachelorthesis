# Bachelorthesis
Python code for a 1D simulation of the Gross-Pitaevskii equation using the first order split step method for the purposes of simulating a full Ramsey sequence. This is all in the theme of measuring gravitational waves using atom interferometry. 
The simulation uses units of \hbar = m = \xi (the healing length of the condensate) = 1
The following is a summation of the methods and functions currently implemented:
Split step methods:
  1. find_ground_state(): Uses imaginary time evolution to approximate the ground state of the BEC for a given potential. This only functions for potentials that stay static through the duration of the calculation.
  2. time_evolution(): Uses real time evolution to evolve the condensate from a given initial condition in time. This method is compatible with dynamic external potentials allowing for example the Ramsey sequence to be simulated (see later lol).
Static potentials:
  1. null_potential(): This is the base case with no external influences. It also functions since the fourier transform in the split step method requires periodicity.
  2. potential_well(): Usual 1D potential well with adjustable height and width.
  3. harmonic_potential(): Usual harmonic potential with adjustable center placement and frequency
  4. potential_well_with_Gauss(): Adds a gaussian bump to the potential well
  5. gravity_potential(): A linearly increasing potential along the x-axis. This is a simple way to simulate a condensate in a 1D column going up, with gravity acting on it. Here the slope is adjustable.
Dynamic potentials: These allow changes through time. Most of these are currently obsolete but were used to develop a correct Ramsey sequence.
  1. delta_spike_potential(): Allows the user to select a time and position to send a delta spike through the condensate.
  2. stirring_potential_Gauss(): Builds on the static potential well with Gaussian bump and moves the bump in a sinusoidal pattern.
  3. Gauss_pulse(): Allows the user to select a time and center of an adjustable Gaussian bump. It has an adjustable spatial width and temporal width. The temporal envelope is also a Gaussian bump.
  4. Gauss_pulse_series(): Similar to before but now allows multiple pulses.
  5. wave_pulse(): Sends a moving wave  for a chosen amplitude, wavelength, velocity, duration and time center of the pulse.
  6. wave_pulse_series(): I mean, pretty self explanatory, no?
  7. interferometer(): Initialises everything to run a full simulation of the wave pulse series potential. Used in testing to find good parameters for a Ramsey sequence.
Visualisation:
1. timeslider_plot(): Visualises the real space with a slider that allows the user to move back and forth through time. The curve shows the modulo squared of the condensate wavefunction |\psi (x)|^2. the colours of the polygon under the curve show the phase of the condensate at that point in space. the other curve visualises the external potential, but this isn't to scale as the units of energy and probability are not the same.
2. reciprocal_timeslider_plot(): Similar to the previous method but shows the reciprocal space. this plot is useful for checking the splitting quality of a Rabi-pulse.
The other methods are general use and will probably be added to this read me later lol. I don't really feel like it right now. Anyways, Idk who tf is reading this, it's a random bachelorthesis. Idk what to put in a README, so hopefully this is something lmao.
