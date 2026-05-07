#Tijdsevolutie van de Gross-Pitaevskii vergelijking in 1D
#Auteur: Xander Scheyltjens
#Laatste update: 31/03/2026

import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import animation
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r"C:\Users\xande\OneDrive\Documents\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

class Gross_Pitaevskii_1D():
    """1D simulation of the Gross-Pitaevskii equation. Includes simulation, plotting and data acquisition methods"""
    #-----Initialisation class----------------------------------------------------------
    def __init__(self, sim_length=50, gridpoints=2**9, dt=125/131072, Natoms=5120):
        self.sim_length = sim_length
        self.gridpoints = gridpoints
        self.dt = dt
        self.Natoms = Natoms

    # Properties for security
    @property
    def sim_length(self):
        return self._sim_length
    @property
    def gridpoints(self):
        return self._gridpoints
    @property
    def dt(self):
        return self._dt
    @property
    def Natoms(self):
        return self._Natoms

    #Add setters with necessary checks
    @sim_length.setter
    def sim_length(self, sim_length):
        if isinstance(sim_length, (int, float)) and sim_length > 0:
            self._sim_length = sim_length
        else:
            raise ValueError('Length of the simulation should be a positive number')
    @gridpoints.setter
    def gridpoints(self, gridpoints):
        if isinstance(gridpoints, int) and gridpoints > 0:
            self._gridpoints = gridpoints
        else:
            raise ValueError('The amount of gridpoints should be a positive integer')
    @dt.setter
    def dt(self, dt):
        if isinstance(dt, (int, float)) and dt > 0:
            self._dt = dt
        else:
            raise ValueError('The time step-size should be a positive number')
    @Natoms.setter
    def Natoms(self, Natoms):
        if isinstance(Natoms, int) and Natoms > 0:
            self._Natoms = Natoms
        else:
            raise ValueError('The amount of atoms should be a positive integer')
    #-----General use methods-------------------------------------------------------------
    def stability_test(self):
        """Tests the stability of the simulation for the given parameters"""
        dx = self.sim_length / self.gridpoints
        if dx >= 1:
            print("Step size should be smaller than the healing length (dx < 1)")
        elif np.log2(self.gridpoints)%1!= 0:
            print("Ideally the number of gridpoints should be a power of 2 to speed up the FFT")
        elif self.Natoms<=self.gridpoints:
            print("For the Gross-Pitaevskii equation to be valid the number of atoms should be much bigger than the number of gridpoints, maybe around 10x")
        elif self.Natoms/self.sim_length<= 1:
            print("The interaction between atoms is too large so the Gross-Pitaevskii equation won't be valid for this system. please increase L or decrease the amount of atoms")
        elif self.dt>=dx**2:
            print("The time step should be smaller than the characteristic time scale, which in our units is dx^2")
        else:
            print("Parameters look good to me :)")

    def initialize_grids(self):
        """Initializes the grids in real and reciprocal space"""
        #Calculate spatial step size
        dx = self.sim_length/self.gridpoints
        #Initialise the grids in real and reciprocal space
        x_grid = np.linspace(-self.sim_length/2,self.sim_length/2,self.gridpoints)
        k_grid = 2*np.pi/dx*fftfreq(self.gridpoints)
        return [x_grid,k_grid]

    def guess_wave_function(self, x_grid):
        """Sets a normalized guess for the wave function for a standard potential well"""
        #Calculate spatial step size
        dx = self.sim_length/self.gridpoints
        #Initialise our guess for the wave function
        psi = -np.tanh((1/4)*(x_grid-self.sim_length/2))*np.tanh((1/4)*(x_grid+self.sim_length/2))
        #Normalize our guess to the number of atoms
        norm = np.trapz(np.abs(psi)**2, dx=dx)
        psi_guess = np.sqrt(self.Natoms/norm)*psi
        return psi_guess

    def harmonic_guess_wave_function(self, x_grid, x_c=0, width=5):
        """Sets a normalized guess for the wave function for a standard potential well"""
        #Calculate spatial step size
        dx = self.sim_length/self.gridpoints
        #Initialise our guess for the wave function
        psi = np.exp(-(x_grid-x_c)**2/(2*width**2))
        #Normalize our guess to the number of atoms
        norm = np.trapz(np.abs(psi)**2, dx=dx)
        psi_guess = np.sqrt(self.Natoms/norm)*psi
        return psi_guess

    def split_beams(self, psi, x_grid, k_kick):
        """Uses operator to impart an idealized momentum kick"""
        psi_new = 1/np.sqrt(2) * (psi * np.exp(1j*k_kick*x_grid)+ psi * np.exp(-1j*k_kick*x_grid))
        return psi_new

    def static_to_dynamic(self, t, V_static):
        """Convert a static 1D potential into a shape that fits the time dependent framework"""
        t_n = int(t / self.dt)
        return np.tile(V_static[:, None], (1, t_n + 1))

    def Ramsey_sequence_generator(self, k_kick, A):
        """Takes in the parameters chosen for the interferometer and returns the correct pulse lengths for a Ramsey sequence"""
        wavelen = [2*np.pi/k_kick, 2*np.pi/k_kick, 2*np.pi/k_kick]
        omega = k_kick**2 / 2
        v = [omega/k_kick, omega/k_kick, omega/k_kick]
        q = [v[0]-0.5*k_kick,v[1]-0.5*k_kick,v[2]-0.5*k_kick]
        pulse_duration = [np.pi/2*(1/np.sqrt((q[0]+k_kick)**2/2-v[0]*k_kick-q[0]**2/2+A**2)),
                         np.pi*(1/np.sqrt((q[1]+k_kick)**2/2-v[1]*k_kick-q[1]**2/2+A**2)),
                         np.pi/2*(1/np.sqrt((q[2]+k_kick)**2/2-v[2]*k_kick-q[2]**2/2+A**2))]
        return wavelen, omega, v, q, pulse_duration

    def integrate(self, state):
        """Integrates a given state in real space"""
        dx = self.sim_length/self.gridpoints
        integral = np.sum(state)*dx
        return integral

    def integrate_reciprocal(self, state):
        """Integrates a given state in reciprocal space"""
        dk = 2*np.pi/self.sim_length
        integral = np.sum(state)*dk
        return integral

    def split_percent(self, k_evo_array, moment, k_kick):
        """Determines the fraction of the condensate transferred to the chosen higher momentum state as a way of determining the quality of the pulse"""
        #We shift the array, mainly to have a more intuitive way of indexing
        k_evo_array = fftshift(k_evo_array, axes=0)
        #We select the correct time at which we want to integrate and get the full norm for normalisation
        index = int(moment/self.dt)
        state = k_evo_array[:,index]
        density_k = np.abs(state)**2
        total = self.integrate_reciprocal(density_k)
        #We find the peak value around the momentum kick, this assures we only grab the peak of the kicked condensate
        dk = 2*np.pi/self.sim_length
        peak_interval_start = int(self.gridpoints//2+k_kick//(2*dk))
        peak_interval_end = int(self.gridpoints//2+3*k_kick//(2*dk))
        #Safety for small k_kick values that result in a length of zero
        if peak_interval_start >= peak_interval_end or len(density_k[peak_interval_start:peak_interval_end]) == 0:
            return float('nan')
        idx_center = peak_interval_start + np.argmax(density_k[peak_interval_start:peak_interval_end])
        #We set a threshold. If the amplitude is lower we consider the peak "done"
        threshold = 0.01 * density_k[idx_center]
        #Set bounds for walking so we don't get as many weird results. We don't want to get an index out of range so set min to 0 and max to -1
        half_width = int(k_kick / (2 * dk))
        if idx_center - half_width >= 0:
            left_bound = idx_center - half_width
        else:
            left_bound = 0
        if idx_center + half_width <= self.gridpoints-1:
            right_bound = idx_center + half_width
        else:
            right_bound = -1
        #Walk left until the amplitude is too low
        left = idx_center
        while left > left_bound and density_k[left] > threshold:
            left -= 1
        #Walk right until amplitude is too low
        right = idx_center
        while right < right_bound and density_k[right] > threshold:
            right += 1
        #Integrate over the array where the peak is located
        split_int = self.integrate_reciprocal(density_k[left:right+1])
        #Calculate what percentage of the condensate was in the peak
        percent = split_int/total*100
        return percent

    def still_percent(self, k_evo_array, moment, k_kick):
        """Determines the fraction of the condensate that has a momentum around zero"""
        #We shift the array, mainly to have a more intuitive way of indexing
        k_evo_array = fftshift(k_evo_array, axes=0)
        #We select the correct time at which we want to integrate and get the full norm for normalisation
        index = int(moment/self.dt)
        state = k_evo_array[:,index]
        density_k = np.abs(state)**2
        total = self.integrate_reciprocal(density_k)
        #We choose to look around index
        dk = 2*np.pi/self.sim_length
        peak_interval_start = int(self.gridpoints//2-k_kick//(2*dk))
        peak_interval_end = int(self.gridpoints//2+k_kick//(2*dk))
        #Safety for small k_kick values that result in a length of zero
        if peak_interval_start >= peak_interval_end or len(density_k[peak_interval_start:peak_interval_end]) == 0:
            return float('nan')
        idx_center = peak_interval_start + np.argmax(density_k[peak_interval_start:peak_interval_end])
        #We set a threshold. If the amplitude is lower we consider the peak "done"
        threshold = 0.01 * density_k[idx_center]
        #Set bounds for walking so we don't get as many weird results. We don't want to get an index out of range so set min to 0 and max to -1
        half_width = int(k_kick / (2 * dk))
        if idx_center - half_width >= 0:
            left_bound = idx_center - half_width
        else:
            left_bound = 0
        if idx_center + half_width <= self.gridpoints-1:
            right_bound = idx_center + half_width
        else:
            right_bound = -1
        #Walk left until the amplitude is too low
        left = idx_center
        while left > left_bound and density_k[left] > threshold:
            left -= 1
        #Walk right until amplitude is too low
        right = idx_center
        while right < right_bound and density_k[right] > threshold:
            right += 1
        #Integrate over the array where the peak is located
        still_int = self.integrate_reciprocal(density_k[left:right+1])
        #Calculate what percentage of the condensate was in the peak
        percent = still_int/total*100
        return percent

    #------Static external potentials------------------------------------------------------
    def null_potential(self, t):
        """Creates a potential matrix that contains no external potential"""
        V = np.zeros(self.gridpoints)
        V = self.static_to_dynamic(t,V)
        return V

    def potential_well(self, t, width = 49, height = 5):
        """Generates a symmetrical potential well of chosen height and width"""
        #Initialize array
        V = np.zeros(self.gridpoints)
        #Calculate index bounds
        leftbound = int(np.floor(self.gridpoints * (0.5 - width/(2*self.sim_length))))
        rightbound = int(np.floor(self.gridpoints * (0.5 + width/(2*self.sim_length))))
        #Check if bounds are in-bounds
        leftbound = max(0, leftbound)
        rightbound = min(self.gridpoints, rightbound)
        #Add external potential outside of bounds
        V[0:leftbound] = height
        V[rightbound:self.gridpoints] = height
        #Make shape of potential matrix fit in our dynamic framework
        V = self.static_to_dynamic(t,V)
        return V

    def harmonic_potential(self, x_grid, t, x_c = 0, omega=1):
        """Generates a harmonic potential well"""
        V = 1/2*omega*(x_grid-x_c)**2
        V = self.static_to_dynamic(t,V)
        return V

    def potential_well_with_Gauss(self, x_grid, t, factor = 0.5, width=49, height=5.0,):
        """Generates a potential well with added Gaussian curve"""
        #Set baseline as potential well
        V_well = self.potential_well(t, width = width, height = height)
        #Add Gaussian bump in the middle
        V = V_well[:,0]+np.exp(-factor*x_grid**2)
        V = self.static_to_dynamic(t,V)
        return V

    def gravity_potential(self, x_grid, t, gravity=0.1):
        """Generates a linearly increasing potential to simulate the gravitational potential"""
        #Set linearly increasing potential
        V = gravity*(x_grid+self.sim_length/2)
        #Set a wall of "infinite" potential
        V[0] = 10**10
        V = self.static_to_dynamic(t,V)
        return V

    #--------Dynamic external potentials---------------------------------------------------
    def delta_spike_potential(self, t, t_spike=None, x_spike=None, A=100):
        """Generates a dynamic potential with delta spikes at chosen position and time"""
        #Set default values if none are given. 
        #We set these seperately since lists are mutable and could cause issues if assigned in the top line
        if t_spike is None:
            t_spike = [1]
        if x_spike is None:
            x_spike = [0]

        #Set baseline as the potential well
        V = self.potential_well(t)
        dx = self.sim_length/self.gridpoints
        #Add spikes at chosen times and places
        for t_s, x_s in zip(t_spike, x_spike):
            x_i = int((x_s-self.sim_length/2)/dx)
            t_i = int(t_s/self.dt)
            V[x_i,t_i] = A
        return V

    def stirring_potential_Gauss(self, t_total, x_grid, width=49, height=5.0,factor=0.5, freq=0.1, A=1.0):
        """ Returns a 2D potential array V[x, t] = box walls + oscillating Gaussian bump. """
        #Create time array
        t_n = int(t_total / self.dt)
        t_arr = np.linspace(0, t_total, t_n+1)
        # Compute the center shift for every time
        shift = 0.2 * self.sim_length * np.sin(2 * np.pi * freq * t_arr)
        # Make X 2D:
        X = x_grid[:, None]
        # Make shift 2D
        S = shift[None, :]
        # Build Gaussian bump
        V = A * np.exp(-factor * (X - S)**2)
        #Add potential well
        V += self.potential_well(t_total, width=width, height=height)
        return V

    def Gauss_pulse(self, x_grid, t, x_c=0, t_c=1.0, pulse_width_x=1, pulse_duration=0.05, A=100):
        """Short Gaussian light pulse."""
        #Create time array
        t_n = int(t / self.dt)
        t_arr = np.linspace(0, t, t_n + 1)
        #Spatial Gaussian
        Gx = np.exp(-(x_grid - x_c)**2 / (2 * pulse_width_x**2))
        #Temporal Gaussian centered at t_s
        envelope = np.exp(-(t_arr - t_c)**2 / (2 * pulse_duration**2))
        #Outer product gives time dependant potential matrix V[x,t]
        V = A * np.outer(Gx, envelope)
        return V

    def Gauss_pulse_series(self, x_grid, t, t_c=None, x_c=None, pulse_duration=0.05, pulse_width_x=1, A=100):
        """Series of Gaussian light pulses."""
        #Set default values if none are given.
        #We set these seperately since lists are mutable and could cause issues if assigned in the top line
        if t_c is None:
            t_c = [1]
        if x_c is None:
            x_c = [0]

        #Create time array
        t_n = int(t / self.dt)
        t_arr = np.linspace(0, t, t_n + 1)
        V = np.zeros((self.gridpoints,t_n+1))
        for t_s, x_s in zip(t_c, x_c):
            Gx = np.exp(-(x_grid - x_s)**2 / (2 * pulse_width_x**2))
            envelope = np.exp(-(t_arr - t_s)**2 / (2 * pulse_duration**2))
            V += A * np.outer(Gx, envelope)
        return V

    def wave_pulse(self, x_grid, t, t_c=1, pulse_duration=0.5, wavelen=1, A=10, v=1):
        """Generates a dynamic potential with a wave pulse at chosen time"""
        #Create time array
        t_n = int(t / self.dt)
        t_arr = np.linspace(0, t, t_n+1)
        #Make the spatial shape of the wave
        wave = np.cos(2*np.pi/wavelen * (x_grid[:,None]-v*t_arr[None,:]))
        envelope = np.where(
        (t_arr >= t_c - pulse_duration/2) &
        (t_arr <  t_c + pulse_duration/2),
        1.0,
        0.0
        )
        #Create time dependant potential
        V = A * wave*envelope[None,:]
        return V

    def wave_pulse_series(self, x_grid, t, t_c=None, pulse_duration=None, wavelen=None, A=10, v=None):
        """Generates a dynamic potential with a wave pulse at chosen time"""
        #Set default values if none are given.
        #We set these seperately since lists are mutable and could cause issues if assigned in the top line
        if t_c is None:
            t_c = [1]
        if pulse_duration is None:
            pulse_duration = [0.5]
        if wavelen is None:
            wavelen = [1]
        if v is None:
            v = [1]

        #Create time array
        t_n = int(t / self.dt)
        V = np.zeros((self.gridpoints,t_n+1))
        for t_s, t_w, wavelen_i, v_i in zip(t_c, pulse_duration, wavelen, v):
            V+=self.wave_pulse(x_grid=x_grid, t=t, t_c=t_s, pulse_duration=t_w, wavelen=wavelen_i, A=A, v=v_i)

        return V

    def interferometer(self, t, t_c=None, pulse_duration=None, wavelen=None, A=10, v=None, k_kick=1, g_factor=-1, plot=True):
        """Complete interferometer simulation without any external phase shift"""
        #Set default values if none are given. 
        #We set these seperately since lists are mutable and could cause issues if assigned in the top line
        if t_c is None:
            t_c = [1]
        if pulse_duration is None:
            pulse_duration = [0.5]
        if wavelen is None:
            wavelen = [1]
        if v is None:
            v = [1]

        x_grid, k_grid = self.initialize_grids()
        psi_guess = self.harmonic_guess_wave_function(x_grid,x_c = -12.5)
        V = self.harmonic_potential(x_grid, t, omega = 0.1, x_c = -12.5)
        [evo_array_ground,_] = self.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-5), nmax = 10**6, g_factor = g_factor)
        ground_state = evo_array_ground[:,-1]
        V = self.potential_well(t, width = 49.9, height = 1000)
        V += self.wave_pulse_series(x_grid,t, t_c, pulse_duration , wavelen, A, v)
        [evo_array, k_evo_array] = self.time_evolution(k_grid, ground_state, V, t, g_factor)
        if plot:
            self.timeslider_plot(x_grid, evo_array, V)
            self.reciprocal_timeslider_plot(k_grid,k_evo_array, k_kick)
        return [evo_array, k_evo_array]

    def interferometer_in_gravity(self, t, t_c=None, pulse_duration=None, wavelen=None, A=10, v=None, k_kick=50, g_factor=-1, plot=True, thesis_plot=False, gravity=0.1):
        """Complete interferometer simulation without any external phase shift"""
        #Set default values if none are given.
        #We set these seperately since lists are mutable and could cause issues if assigned in the top line
        if t_c is None:
            t_c = [1]
        if pulse_duration is None:
            pulse_duration = [0.5]
        if wavelen is None:
            wavelen = [1]
        if v is None:
            v = [1]

        x_grid, k_grid = self.initialize_grids()
        psi_guess = self.harmonic_guess_wave_function(x_grid,x_c = -12.5)
        #V = self.gravity_potential(x_grid, t)
        V = self.harmonic_potential(x_grid, t, omega = 0.1, x_c = -12.5)
        [evo_array_ground,_] = self.find_ground_state(k_grid, psi_guess, V, nmax = 10**6, g_factor = g_factor)
        ground_state = evo_array_ground[:,-1]
        V = self.potential_well(t, width = 49.9, height = 1000)
        V += self.gravity_potential(x_grid, t, gravity)
        V += self.wave_pulse_series(x_grid,t, t_c, pulse_duration , wavelen, A, v)
        [evo_array, k_evo_array] = self.time_evolution(k_grid, ground_state, V, t, g_factor)
        if plot:
            self.timeslider_plot(x_grid, evo_array, V)
            self.reciprocal_timeslider_plot(k_grid,k_evo_array, k_kick, Lorentz=True)
        if thesis_plot:
            self.time_slices_plot(t, x_grid, evo_array, V, moments=[0, 0.1, 0.8, 1.5, 2.2, 2.9, 3]) #moments for gravity: [0, 0.1, 0.5, 0.9, 1.3, 1.7, 2.4] moments other plots:[0, 0.1, 0.8, 1.5, 2.2, 2.9, 3]
            self.reciprocal_time_slices_plot(t, k_grid, k_evo_array, k_kick, moments=[0, 0.1, 0.8, 1.5, 2.2, 2.9, 3])
            self.animate_reciprocal_evolution(k_grid, k_evo_array, k_kick, filename="gravity_reciprocal_final.mp4", fps=30)
        return [evo_array, k_evo_array]

    #-------Split step methods--------------------------------------------------------------
    def find_ground_state(self, k_grid, psi_guess, V, TOL=10**(-5), nmax=10**4, g_factor=-1):
        """Uses the split step method with an imaginary time evolution to relax the wavefunction toward the ground state"""
        g = self.sim_length/self.Natoms
        dx = self.sim_length/self.gridpoints
        counter = 0
        #Initialise the kinetic evolution operator which is constant through the loop
        kin_evo = np.exp(-1/4*(k_grid**2)*self.dt)
        psi = psi_guess
        FFT_psi = fft(psi)
        evo_array = np.zeros((self.gridpoints, nmax + 2), dtype=complex)
        k_evo_array = np.zeros((self.gridpoints, nmax + 2), dtype=complex)
        evo_array[:,0] = psi_guess
        k_evo_array[:,0] = fft(psi_guess)
        error = 1
        #Loop over time evolution for set amount of steps
        while error > TOL and counter <= nmax:
            psi_old = psi
            FFT_psi *= kin_evo
            psi = ifft(FFT_psi)
            density = np.absolute(psi)**2
            pot_evo = np.exp(-(V[:,0]-g_factor*g*density-1)*self.dt) #Here we use mu=\pm1 in our units
            psi *= pot_evo
            FFT_psi = fft(psi)
            FFT_psi *= kin_evo
            psi = ifft(FFT_psi)
            norm_psi = np.trapz(np.abs(psi)**2, dx=dx)
            psi *= np.sqrt(self.Natoms/norm_psi)
            error = np.max(abs(psi_old-psi))
            counter+=1
            evo_array[:,counter] = psi
            k_evo_array[:,counter] = FFT_psi
        evo_array = evo_array[:,0:counter]
        k_evo_array = k_evo_array[:,0:counter]
        if counter>= nmax:
            print("Given tolerance not reached, simulation stopped after ", nmax, " loops")
        return [evo_array, k_evo_array]

    def time_evolution(self, k_grid, psi_guess, V, t, g_factor=-1):
        """Uses the split step method to evolve the Gross-Pitaevskii equation over time"""
        g = self.sim_length/self.Natoms
        i = 0+1j
        time = 0
        counter = 0
        #Initialise the kinetic evolution operator which is constant through the loop
        kin_evo = np.exp(-i/4*(k_grid**2)*self.dt)
        psi = psi_guess.astype(np.complex128)
        FFT_psi = fft(psi)
        n_steps = int(t / self.dt)
        evo_array = np.zeros((self.gridpoints, n_steps + 2), dtype=complex)
        k_evo_array = np.zeros((self.gridpoints, n_steps + 2), dtype=complex)
        evo_array[:,0] = psi_guess
        k_evo_array[:,0] = fft(psi_guess)
        #Loop over time evolution for set amount of steps
        while time<t:
            FFT_psi *= kin_evo
            psi = ifft(FFT_psi)
            density = np.abs(psi)**2
            pot_evo = np.exp(-i*(V[:,counter]-g_factor*g*density)*self.dt)
            psi *= pot_evo
            FFT_psi = fft(psi)
            FFT_psi *= kin_evo
            psi = ifft(FFT_psi)
            time+=self.dt
            counter += 1
            evo_array[:,counter] = psi
            k_evo_array[:,counter] = FFT_psi
        return [evo_array, k_evo_array]

    #------- Visualization -----------------------------------------------------------------
    def Lorentz_curve(self, ax, A=34, m=129):
        dk = 2*np.pi/self.sim_length 
        k_kick = m*dk
        omega = k_kick**2 / 2
        v = omega/k_kick
        q_res = k_kick
        dx = self.sim_length/self.gridpoints
        q = np.linspace(-2*np.pi/dx, 2*np.pi/dx, 1000)
        omega_10 = (1/2) * (2*(q-q_res)*k_kick + k_kick**2) - v*k_kick

        lorentz = A**2 / (A**2 + omega_10**2)

        line, = ax.plot(q, lorentz, label="Lorentzian", color='k', linewidth=1)
        return line

    def timeslider_plot(self, x_grid, evo_array, V):
        _, n_times = evo_array.shape

        # ---- Precomputation -------------------------------------------------
        # Global max density (vertical scale stays constant)
        global_ymax = np.max(np.abs(evo_array)**2)
        y_min, y_max = 0, global_ymax
        # Precompute density and wrapped phases for all timesteps
        densities = np.abs(evo_array)**2
        phases = np.angle(evo_array)
        phases = (phases + np.pi) % (2 * np.pi) - np.pi #Doing it this way fixes a visual glitch
        # Precompute polygons for all timesteps
        all_polys = []
        for t in range(n_times):
            verts = [(x_grid[0], y_min)] + list(zip(x_grid, densities[:, t])) + [(x_grid[-1], y_min)]
            poly = Polygon(verts, facecolor='none', edgecolor='none')
            all_polys.append(poly)
        # Precompute vertical grid for phase image
        Y = np.linspace(y_min, y_max, 400)
        # Initial phase-gradient image (first timestep)
        Z = np.tile(phases[:, 0], (len(Y), 1)).astype(np.float32)

        # ---- Plot setup --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))
        ax2 = ax.twinx()
        #plt.style.use(['science', 'no-latex'])
        fig.subplots_adjust(bottom=0.25, right=0.9)
        # Initial lines
        line, = ax.plot(x_grid, densities[:, 0], lw=1)
        line2, = ax2.plot(x_grid, V[:,0], lw=1, color='orange')
        V_max = np.max(V[10:-10])
        #Set labels
        ax2.set_ylim(0, V_max*10)
        ax.set_xlabel("Positie")
        ax.set_ylabel('$|\psi(x,t)|^2$')
        ax2.set_ylabel('Externe potentiaal')
        # Initial phase image
        im = ax.imshow(
            Z, extent=[x_grid.min(), x_grid.max(), y_min, y_max+0.1*y_max],
            origin='lower', cmap='twilight', aspect='auto', vmin=-np.pi, vmax=np.pi
        )
        # Add initial polygon clip
        poly = all_polys[0]
        ax.add_patch(poly)
        im.set_clip_path(poly)
        #Add colorbar
        fig.colorbar(im, ax=[ax, ax2], label='Fase', )
        #Create slider
        axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        time_slider = plt.Slider(ax=axtime, label='Tijd',
                                  valmin=0, valmax=(n_times-1)*self.dt, valinit=0)
        # ---- Update function ----------------------------------------------------
        def update(_):
            t = int(time_slider.val/self.dt)
            # Update density line
            line.set_ydata(densities[:, t])
            line2.set_ydata(V[:,t-1])
            # Update polygon clip
            poly.set_xy(all_polys[t].get_xy())
            # Update phase image
            Z[:] = phases[:, t]
            im.set_data(Z)
            fig.canvas.draw_idle()

        time_slider.on_changed(update)

        plt.show()

    def reciprocal_timeslider_plot(self, k_grid, k_evo_array, k_kick, Lorentz=False):
        _, n_times = k_evo_array.shape

        # ---- Precomputation -------------------------------------------------
        # Shift the frequencies to look normal
        k_evo_array = fftshift(k_evo_array, axes=0)
        k_grid = fftshift(k_grid)
        # Global max density (vertical scale stays constant)
        global_ymax = np.max(np.abs(k_evo_array)**2)
        y_min, y_max = 0, global_ymax
        # Precompute density and wrapped phases for all timesteps
        densities = np.abs(k_evo_array)**2
        phases = np.angle(k_evo_array)
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
        # Precompute polygons for all timesteps
        all_polys = []
        for t in range(n_times):
            verts = [(k_grid[0], y_min)] + list(zip(k_grid, densities[:, t])) + [(k_grid[-1], y_min)]
            poly = Polygon(verts, facecolor='none', edgecolor='none')
            all_polys.append(poly)
        # Precompute vertical grid for phase image
        Y = np.linspace(y_min, y_max, 400)
        # Initial phase-gradient image (first timestep)
        Z = np.tile(phases[:, 0], (len(Y), 1)).astype(np.float32)

        # ---- Plot setup --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.subplots_adjust(bottom=0.25)

        # Create Brillouin zone background
        k_min, k_max = -2*k_kick, 2*k_kick
        # Colors
        color_1 = '#fff7bc'  # 1st BZ (pastel yellow)
        color_2 = '#c6dbef'  # 2nd BZ (pastel blue)
        color_3 = '#ff6961'  # 3rd BZ (pastel red)
        # 1st BZ
        ax.axvspan(-k_kick/2, k_kick/2, color=color_1, alpha=0.3, zorder=0)
        # 2nd BZ (left and right)
        ax.axvspan(-3*k_kick/2, -k_kick/2, color=color_2, alpha=0.3, zorder=0)
        ax.axvspan(k_kick/2, 3*k_kick/2, color=color_2, alpha=0.3, zorder=0)
        # 3rd BZ (only visible halves within plotting range)
        ax.axvspan(k_min, -3*k_kick/2, color=color_3, alpha=0.3, zorder=0)
        ax.axvspan(3*k_kick/2, k_max, color=color_3, alpha=0.3, zorder=0)

        y_top = y_max
        ax.text(0, y_top, f"$n=0$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        ax.text(k_kick, y_top, f"$n=1$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        ax.text(-k_kick, y_top, f"$n=-1$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        #Mark the edges of the BZ
        for n in range(-3, 4):
            boundary = (n + 0.5) * k_kick
            ax.axvline(boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Initial density line
        line, = ax.plot(k_grid, densities[:, 0], lw=1)
        if Lorentz:
            ax2 = ax.twinx()
            line2 = self.Lorentz_curve(ax2)
            plt.title("Reciproke ruimte met Lorentzcurve rond $k_{kick}$") #was created for a single plot in the report
            ax2.set_ylim(0,1.2)
            ax2.set_ylabel("Fractie overgedragen atomen")
        # Set labels
        ax.set_xlabel("Impuls")
        ax.set_ylabel('$|\psi(k,t)|^2$')
        # Initial phase image
        im = ax.imshow(
            Z, extent=[-2*k_kick, 2*k_kick, y_min, y_max+0.1*y_max],
            origin='lower', cmap='twilight', aspect='auto', vmin=-np.pi, vmax=np.pi
        )
        # Add initial polygon clip
        poly = all_polys[0]
        ax.add_patch(poly)
        im.set_clip_path(poly)
        #Add colorbar
        if Lorentz:
            fig.colorbar(im, ax=[ax,ax2], label='Fase')
        else:
            fig.colorbar(im, ax=ax, label='Fase')
        #Create slider
        axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        time_slider = plt.Slider(ax=axtime, label='Tijd',
                                  valmin=0, valmax=(n_times-1)*self.dt, valinit=0)

        # ---- Update function ----------------------------------------------------
        def update(_):
            t = int(time_slider.val/self.dt)
            # Update density line
            line.set_ydata(densities[:, t])
            # Update polygon clip
            poly.set_xy(all_polys[t].get_xy())
            # Update phase image
            Z[:] = phases[:, t]
            im.set_data(Z)

            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()

    def time_slices_plot(self, t, x_grid, evo_array, V, moments=None, n_slices=6):
        densities = np.abs(evo_array)**2
        phases = np.angle(evo_array)
        phases = (phases + np.pi) % (2 * np.pi) - np.pi

        # ---- Choose time slices ----
        if moments is None:
            moments = np.linspace(0, t, n_slices, dtype=float)
        else:
            moments = np.array(moments, dtype=float)

        # ---- Normalization & offset ----
        max_density = np.max(densities)
        offset = 0.8*max_density  # spacing between curves

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        for i, ti in enumerate(moments/self.dt):
            y_offset = i * offset
            density = densities[:, int(ti)]
            potential = V[1:-2,int(ti)]
            potential_scaled = potential / np.max(V[1:-2]) * 0.1*max_density
            # Shifted density
            y_vals = density + y_offset
            shift_potential = potential_scaled + y_offset

            # Plot line
            ax.plot(x_grid, y_vals, color='black', lw=1)
            ax.plot(x_grid[1:-2], shift_potential, lw=1, color='orange')

            # ---- Phase colormap fill (your idea preserved) ----
            verts = [(x_grid[0], y_offset)] + list(zip(x_grid, y_vals)) + [(x_grid[-1], y_offset)]
            poly = Polygon(verts, facecolor='none', edgecolor='none')
            poly.set_zorder(1.5)
            ax.add_patch(poly)

            # Create phase image
            Y = np.linspace(y_offset, y_offset + offset, 200)
            Z = np.tile(phases[:, int(ti)], (len(Y), 1))

            im = ax.imshow(
                Z,
                extent=[x_grid.min(), x_grid.max(), y_offset, y_offset + offset],
                origin='lower',
                cmap='twilight',
                aspect='auto',
                vmin=-np.pi,
                vmax=np.pi,
                zorder=1  # <-- important
            )

            ax.plot(x_grid, y_vals, color='black', lw=1, zorder=2)
            im.set_clip_path(poly)

            # Optional: label time
            ax.text(x_grid.max(), y_offset, f"t = {ti*self.dt:.2f}",
                    va='bottom', ha='right', fontsize=8)

        ax.set_xlabel("Positie")

        # Clean up look (important for thesis visuals)
        ax.set_yticks([])
        ax.spines[['left', 'right', 'top']].set_visible(False)

        plt.colorbar(im, ax=ax, label='Fase')

        plt.show()

    def reciprocal_time_slices_plot(self, t, k_grid, k_evo_array, k_kick, moments=None, n_slices=6):
        # Shift the frequencies to look normal
        k_evo_array = fftshift(k_evo_array, axes=0)
        k_grid = fftshift(k_grid)
        #Same computations as always
        densities = np.abs(k_evo_array)**2
        phases = np.angle(k_evo_array)
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
        #For the spacing of the plots later. Doing it now so check is easier
        max_density = np.max(densities)
        offset = 0.8*max_density  # spacing between curves
        if moments is None:
            y_top = n_slices * offset+1.1*densities[-1]
        else:
            y_top = (len(moments)-1) * offset+1.1*np.max(densities[:, int(moments[-1]/self.dt)])
        # ---- Choose time slices ----
        if moments is None:
            moments = np.linspace(0, t, n_slices, dtype=float)
        else:
            moments = np.array(moments, dtype=float)

        # ---- Normalization & offset ----
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        # Create Brillouin zone background
        k_min, k_max = -2*k_kick, 2*k_kick
        # Colors
        color_1 = '#fff7bc'  # 1st BZ (pastel yellow)
        color_2 = '#c6dbef'  # 2nd BZ (pastel blue)
        color_3 = '#ff6961'  # 3rd BZ (pastel red)
        # 1st BZ
        ax.axvspan(-k_kick/2, k_kick/2, color=color_1, alpha=0.3, zorder=0)
        # 2nd BZ 
        ax.axvspan(-3*k_kick/2, -k_kick/2, color=color_2, alpha=0.3, zorder=0)
        ax.axvspan(k_kick/2, 3*k_kick/2, color=color_2, alpha=0.3, zorder=0)
        # 3rd BZ 
        ax.axvspan(k_min, -3*k_kick/2, color=color_3, alpha=0.3, zorder=0)
        ax.axvspan(3*k_kick/2, k_max, color=color_3, alpha=0.3, zorder=0)
        #Add labels
        ax.text(0, y_top, f"$n=0$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        ax.text(k_kick, y_top, f"$n=1$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        ax.text(-k_kick, y_top, f"$n=-1$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        #Mark the edges of the BZ
        for n in range(-3, 4):
            boundary = (n + 0.5) * k_kick
            ax.axvline(boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Plot all of the different moments in a joyplot like fashion
        for i, t in enumerate(moments/self.dt):
            y_offset = i * offset
            density = densities[:, int(t)]

            # Shifted density
            y_vals = density + y_offset

            # Plot line
            ax.plot(k_grid, y_vals, color='black', lw=1)

            # ---- Phase colormap fill ----
            verts = [(k_grid[0], y_offset)] + list(zip(k_grid, y_vals)) + [(k_grid[-1], y_offset)]
            poly = Polygon(verts, facecolor='none', edgecolor='none')
            poly.set_zorder(1.5)
            ax.add_patch(poly)

            # Create phase image
            Y = np.linspace(y_offset, y_offset + offset, 200)
            Z = np.tile(phases[:, int(t)], (len(Y), 1))

            im = ax.imshow(
                Z,
                extent=[-2*k_kick, 2*k_kick, y_offset, y_offset + offset],
                origin='lower',
                cmap='twilight',
                aspect='auto',
                vmin=-np.pi,
                vmax=np.pi,
                zorder=1  # <-- important
            )

            ax.plot(k_grid, y_vals, color='black', lw=1, zorder=2)
            im.set_clip_path(poly)
            ax.set_xlim(-2*k_kick, 2*k_kick)
            # Label time
            ax.text(k_grid.max(), y_offset, f"t = {t*self.dt:.2f}",
                    va='bottom', ha='right', fontsize=8)

        ax.set_xlabel("Impuls")

        # Clean up look
        ax.set_yticks([])
        ax.spines[['left', 'right', 'top']].set_visible(False)

        plt.colorbar(im, ax=ax, label='Fase')

        plt.show()

    def animate_evolution(self, x_grid, evo_array, V, filename="animation.mp4", fps=30):
        _, n_times = evo_array.shape

        # ---- Precomputation -------------------------------------------------
        densities = np.abs(evo_array)**2
        phases = np.angle(evo_array)
        phases = (phases + np.pi) % (2 * np.pi) - np.pi

        global_ymax = np.max(densities)
        y_min, y_max = 0, global_ymax

        # Precompute polygons
        all_polys = []
        for t in range(n_times):
            verts = [(x_grid[0], y_min)] + list(zip(x_grid, densities[:, t])) + [(x_grid[-1], y_min)]
            all_polys.append(verts)

        # Phase image grid
        Y = np.linspace(y_min, y_max, 400)
        Z = np.tile(phases[:, 0], (len(Y), 1)).astype(np.float32)

        # ---- Plot setup -----------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))
        ax2 = ax.twinx()

        line, = ax.plot(x_grid, densities[:, 0], lw=1)
        line2, = ax2.plot(x_grid, V[:, 0], lw=1, color='orange')

        V_max = np.max(V[10:-10])
        ax2.set_ylim(0, V_max * 10)

        ax.set_xlabel("Positie")
        ax.set_ylabel(r'$|\psi(x,t)|^2$')
        ax2.set_ylabel('Externe potentiaal')

        im = ax.imshow(
            Z,
            extent=[x_grid.min(), x_grid.max(), y_min, y_max + 0.1*y_max],
            origin='lower',
            cmap='twilight',
            aspect='auto',
            vmin=-np.pi,
            vmax=np.pi
        )

        poly = Polygon(all_polys[0], facecolor='none', edgecolor='none')
        ax.add_patch(poly)
        im.set_clip_path(poly)

        fig.colorbar(im, ax=[ax, ax2], label='Fase')

        # ---- Animation update ----------------------------------------------
        def update(frame):
            t = frame

            line.set_ydata(densities[:, t])
            line2.set_ydata(V[:, t-1] if t > 0 else V[:, 0])

            # Update polygon
            poly.set_xy(all_polys[t])

            # Update phase image
            Z[:] = phases[:, t]
            im.set_data(Z)

            return line, line2, im, poly

        step = 20
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=range(0,n_times,step),
            interval=1000/fps,
            blit=True
        )
        print(n_times)
        # ---- Save -----------------------------------------------------------
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(filename, writer=writer, dpi=300)

        print(f"Saved animation to {filename}")

    def animate_reciprocal_evolution(self, k_grid, k_evo_array, k_kick, filename="animation.mp4", fps=30):
        _, n_times = k_evo_array.shape

        # ---- Precomputation -------------------------------------------------
        # Shift the frequencies to look normal
        k_evo_array = fftshift(k_evo_array, axes=0)
        k_grid = fftshift(k_grid)
        # Global max density (vertical scale stays constant)
        global_ymax = np.max(np.abs(k_evo_array)**2)
        y_min, y_max = 0, global_ymax
        # Precompute density and wrapped phases for all timesteps
        densities = np.abs(k_evo_array)**2
        phases = np.angle(k_evo_array)
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
        # Precompute polygons for all timesteps
        all_polys = []
        for t in range(n_times):
            verts = [(k_grid[0], y_min)] + list(zip(k_grid, densities[:, t])) + [(k_grid[-1], y_min)]
            all_polys.append(verts)
        # Precompute vertical grid for phase image
        Y = np.linspace(y_min, y_max, 400)
        # Initial phase-gradient image (first timestep)
        Z = np.tile(phases[:, 0], (len(Y), 1)).astype(np.float32)

        # ---- Plot setup --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.subplots_adjust(bottom=0.25)

        # Create Brillouin zone background
        k_min, k_max = -2*k_kick, 2*k_kick
        # Colors
        color_1 = '#fff7bc'  # 1st BZ (pastel yellow)
        color_2 = '#c6dbef'  # 2nd BZ (pastel blue)
        color_3 = '#ff6961'  # 3rd BZ (pastel red)
        # 1st BZ
        ax.axvspan(-k_kick/2, k_kick/2, color=color_1, alpha=0.3, zorder=0)
        # 2nd BZ (left and right)
        ax.axvspan(-3*k_kick/2, -k_kick/2, color=color_2, alpha=0.3, zorder=0)
        ax.axvspan(k_kick/2, 3*k_kick/2, color=color_2, alpha=0.3, zorder=0)
        # 3rd BZ (only visible halves within plotting range)
        ax.axvspan(k_min, -3*k_kick/2, color=color_3, alpha=0.3, zorder=0)
        ax.axvspan(3*k_kick/2, k_max, color=color_3, alpha=0.3, zorder=0)

        y_top = y_max
        ax.text(0, y_top, f"$n=0$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        ax.text(k_kick, y_top, f"$n=1$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        ax.text(-k_kick, y_top, f"$n=-1$", ha='center', va='bottom', fontsize=10, alpha=0.7)
        #Mark the edges of the BZ
        for n in range(-3, 4):
            boundary = (n + 0.5) * k_kick
            ax.axvline(boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Initial density line
        line, = ax.plot(k_grid, densities[:, 0], lw=1)
        # Set labels
        ax.set_xlabel("Impuls")
        ax.set_ylabel('$|\psi(k,t)|^2$')
        # Initial phase image
        im = ax.imshow(
            Z, extent=[-2*k_kick, 2*k_kick, y_min, y_max+0.1*y_max],
            origin='lower', cmap='twilight', aspect='auto', vmin=-np.pi, vmax=np.pi
        )
        # Add initial polygon clip
        poly = Polygon(all_polys[0], facecolor='none', edgecolor='none')
        ax.add_patch(poly)
        im.set_clip_path(poly)

        fig.colorbar(im, ax=ax, label='Fase')
        # ---- Animation update ----------------------------------------------
        def update(frame):
            t = frame

            line.set_ydata(densities[:, t])

            # Update polygon
            poly.set_xy(all_polys[t])

            # Update phase image
            Z[:] = phases[:, t]
            im.set_data(Z)

            return line, im, poly

        step = 20
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=range(0,n_times,step),
            interval=1000/fps,
            blit=True
        )
        print(n_times)
        # ---- Save -----------------------------------------------------------
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(filename, writer=writer, dpi=300)

        print(f"Saved animation to {filename}")