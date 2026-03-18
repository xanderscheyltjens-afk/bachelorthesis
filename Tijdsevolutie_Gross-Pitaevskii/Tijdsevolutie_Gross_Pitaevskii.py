#Tijdsevolutie van de Gross-Pitaevskii vergelijking in 1D
#Auteur: Xander Scheyltjens
#Laatste update: 2/12/2025

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class Gross_Pitaevskii_1D():
    def __init__(self, L = 50, n = 2**9, dt = 125/131072, Natoms = 5120):
        self.L = L
        self.n = n
        self.dt = dt
        self.Natoms = Natoms

    # Properties for security
    @property
    def L(self): return self._L
    @property
    def n(self): return self._n
    @property
    def dt(self): return self._dt
    @property
    def Natoms(self): return self._Natoms

    #Add setters with necessary checks
    @L.setter
    def L(self, L):
        if isinstance(L, (int, float)) and L > 0:
            self._L = L
        else:
            raise ValueError('Length of the simulation should be a positive number') 
    @n.setter
    def n(self, n):
        if isinstance(n, int) and n > 0:
            self._n = n
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
   
    #-----General use methods--------------------------------------------------------------
    def stability_test(self):
        """Tests the stability of the simulation for the given parameters"""
        dx = self.L / self.n
        if dx >= 1:
            print("Step size should be smaller than the healing length (dx < 1)")
        elif np.log2(self.n)%1!= 0:
            print("Ideally the number of gridpoints should be a power of 2 to speed up the FFT")
        elif self.Natoms<=self.n:
            print("For the Gross-Pitaevskii equation to be valid the number of atoms should be much bigger than the number of gridpoints, maybe around 10x")
        elif self.Natoms/self.L<= 1:
            print("The interaction between atoms is too large so the Gross-Pitaevskii equation won't be valid for this system. please increase L or decrease the amount of atoms")
        elif self.dt>=(self.L/self.n)**2:
            print("The time step should be smaller than the characteristic time scale, which in our units is dx^2")
        else:
            print("Parameters look good to me :)")

    def initialize_grids(self):
        """Initializes the grids in real and reciprocal space"""
        #Calculate spatial step size
        dx = self.L/self.n
        #Initialise the grids in real and reciprocal space
        x_grid = np.linspace(-self.L/2,self.L/2,self.n)
        k_grid = 2*np.pi/dx*fftfreq(self.n)
        return [x_grid,k_grid]
                     
    def guess_wave_function(self,x_grid):
        """Sets a normalized guess for the wave function"""
        #Calculate spatial step size
        dx = self.L/self.n
        #Initialise our guess for the wave function
        psi_0 = -np.tanh((1/4)*(x_grid-self.L/2))*np.tanh((1/4)*(x_grid+self.L/2))
        #Normalize our guess to the number of atoms
        norm_0 = np.trapz(np.abs(psi_0)**2, dx=dx)
        psi_guess = np.sqrt(self.Natoms/norm_0)*psi_0
        return psi_guess

    def split_beams(self, psi, x_grid, k):
        """Uses operator to impart an idealized momentum kick"""
        psi_new = 1/np.sqrt(2) * (psi * np.exp(1j*k*x_grid)+ psi * np.exp(-1j*k*x_grid))
        return psi_new

    def static_to_dynamic(self,t, V_static):
        """Convert a static 1D potential into a shape that fits the time dependent framework"""
        t_n = int(t / self.dt)
        return np.tile(V_static[:, None], (1, t_n + 1))

    #------Static external potentials-------------------------------------------------
    def null_potential(self,t):
        """Creates a potential matrix that contains no external potential"""
        V = np.zeros(self.n)
        V = self.static_to_dynamic(t,V)
        return V

    def potential_well(self,t,width = 49, height = 5):
        """Generates a symmetrical potential well of chosen height and width"""
        #Initialize array
        V = np.zeros(self.n)
        #Calculate index bounds
        leftbound = int(np.floor(self.n * (0.5 - width/(2*self.L))))
        rightbound = int(np.floor(self.n * (0.5 + width/(2*self.L))))
        #Check if bounds are in-bounds
        leftbound = max(0, leftbound)
        rightbound = min(self.n, rightbound)
        #Add external potential outside of bounds
        V[0:leftbound] = height
        V[rightbound:self.n] = height
        #Make shape of potential matrix fit in our dynamic framework
        V = self.static_to_dynamic(t,V)
        return V

    def harmonic_potential(self,x_grid,t,x_c = 0,omega=1):
        """Generates a harmonic potential well"""
        V = 1/2*omega*(x_grid-x_c)**2
        V = self.static_to_dynamic(t,V)
        return V

    def potential_well_with_Gauss(self, x_grid,t, factor = 0.5, width=49, height=5.0,):
        """Generates a potential well with added Gaussian curve"""
        #Set baseline as potential well
        V_well = self.potential_well(t, width = width, height = height)
        #Add Gaussian bump in the middle
        V = V_well[:,0]+np.exp(-factor*x_grid**2)
        V = self.static_to_dynamic(t,V)
        return V

    def gravity_potential(self,x_grid,t, rico=0.1):
        """Generates a linearly increasing potential to simulate the gravitational potential"""
        #Set linearly increasing potential
        V = rico*(x_grid+self.L/2)
        #Set a wall of "infinite" potential
        V[0] = 10**10
        V = self.static_to_dynamic(t,V)
        return V

    #--------Dynamic external potentials-----------------------------------------------------
    def delta_spike_potential(self, t, t_spike = [1], x_spike = [0], A = 100):
        """Generates a dynamic potential with delta spikes at chosen position and time"""
        #Set baseline as the potential well
        V = self.potential_well(t)
        dx = self.L/self.n
        #Add spikes at chosen times and places
        for t_s, x_s in zip(t_spike, x_spike):
            x_i = int((x_s-self.L/2)/dx)
            t_i = int(t_s/self.dt)
            V[x_i,t_i] = A 
        return V

    def stirring_potential_Gauss(self, t_total, x_grid, width=49, height=5.0,factor=0.5, freq=0.1, amp=1.0):
        """ Returns a 2D potential array V[x, t] = box walls + oscillating Gaussian bump. """
        #Create time array
        t_n = int(t_total / self.dt)
        t_arr = np.linspace(0, t_total, t_n+1)
        # Compute the center shift for every time
        shift = 0.2 * self.L * np.sin(2 * np.pi * freq * t_arr)
        # Make X 2D: 
        X = x_grid[:, None]
        # Make shift 2D
        S = shift[None, :]
        # Build Gaussian bump
        V = amp * np.exp(-factor * (X - S)**2)
        #Add potential well
        V += self.potential_well(t_total, width=width, height=height)
        return V

    def Gauss_pulse(self, x_grid, t, x_c=0, t_c=1.0, pulse_width_x=1,pulse_width_t=0.05,A=100):
        """Short Gaussian light pulse."""
        #Create time array
        t_n = int(t / self.dt)
        t_arr = np.linspace(0, t, t_n + 1)
        #Spatial Gaussian
        Gx = np.exp(-(x_grid - x_c)**2 / (2 * pulse_width_x**2))
        #Temporal Gaussian centered at t_s
        envelope = np.exp(-(t_arr - t_c)**2 / (2 * pulse_width_t**2))
        #Outer product gives time dependant potential matrix V[x,t]
        V = A * np.outer(Gx, envelope)
        return V

    def Gauss_pulse_series(self, x_grid, t, x_c=[0], t_c=[1.0], pulse_width_x=1,pulse_width_t=0.05,A=100):
        """Series of Gaussian light pulses."""
        #Create time array
        t_n = int(t / self.dt)
        t_arr = np.linspace(0, t, t_n + 1)
        V = np.zeros((self.n,t_n+1))
        for t_s, x_s in zip(t_c, x_c):
            Gx = np.exp(-(x_grid - x_s)**2 / (2 * pulse_width_x**2))
            envelope = np.exp(-(t_arr - t_s)**2 / (2 * pulse_width_t**2))
            V += A * np.outer(Gx, envelope)
        return V

    def wave_pulse(self, x_grid, t, t_c = 1, pulse_width_t = 0.5, wavelen = 1, A = 10, v = 1, Gauss = None):
        """Generates a dynamic potential with a wave pulse at chosen time"""
        #Create time array
        t_n = int(t / self.dt)
        t_arr = np.linspace(0, t, t_n+1)
        #Make the spatial shape of the wave
        wave = np.cos(2*np.pi/wavelen * (x_grid[:,None]-v*t_arr[None,:]))
        #Make temporal envelope for the pulse (Gaussian)
        if Gauss != None:
            envelope = np.exp(-(t_arr - t_c)**2 / (2*pulse_width_t**2))
        else:
            envelope = np.where(
            (t_arr >= t_c - pulse_width_t/2) &
            (t_arr <  t_c + pulse_width_t/2),
            1.0,
            0.0
            )
        #Create time dependant potential
        V = A * wave*envelope[None,:]
        return V

    def wave_pulse_series(self, x_grid, t, t_c = [1], pulse_width_t = [0.5], wavelen = [1], A = 10, v = [1], Gauss = None, dk = 1):
        """Generates a dynamic potential with a wave pulse at chosen time"""
        t_n = int(t / self.dt)
        t_arr = np.linspace(0, t, t_n+1)
        V = np.zeros((self.n,t_n+1))
        for t_s, t_w, wavelen, v in zip(t_c,pulse_width_t,wavelen, v):
           V+=self.wave_pulse(x_grid, t, t_s, t_w, wavelen, A, v, Gauss)
        return V

    def interferometer(self, t, t_c = [1], pulse_width_t = [0.5], wavelen = [1], A = 10, v = [1], Gauss=None):
        """Complete interferometer simulation without any external phase shift"""
        x_grid, k_grid = self.initialize_grids()
        psi_guess = self.guess_wave_function(x_grid)
        V = self.harmonic_potential(x_grid, t, omega = 0.1, x_c = -12.5)
        evo_array_ground = self.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-4), nmax = 10**6)
        ground_state = evo_array_ground[:,evo_array_ground.shape[1]-1]
        V = self.potential_well(t, width = 49.9, height = 1000)
        V += self.wave_pulse_series(x_grid,t, t_c, pulse_width_t , wavelen, A, v, Gauss)
        evo_array = self.time_evolution(k_grid, ground_state, V, t)
        self.timeslider_plot(x_grid, evo_array, V)

    #-------Split step methods---------------------------------------------------------------
    def find_ground_state(self, k_grid, psi_guess, V, TOL=10**(-5), nmax = 10**4, sign = -1):
        """Uses the split step method with an imaginary time evolution to relax the wavefunction toward the ground state"""
        g = self.L/self.Natoms
        dx = self.L/self.n
        n = 0
        #Initialise the kinetic evolution operator which is constant through the loop
        kin_evo = np.exp(-1/4*(k_grid**2)*self.dt)
        psi = psi_guess
        evo_array = np.zeros((self.n, nmax + 2), dtype=complex)
        evo_array[:,0] = psi_guess
        error = 1
        #Loop over time evolution for set amount of steps
        while error > TOL and n <= nmax:
            psi_old = psi
            FFT_psi = fft(psi)
            FFT_psi *= kin_evo
            psi = ifft(FFT_psi)
            prob_dist = np.absolute(psi_old)**2
            pot_evo = np.exp(-(V[:,0]-sign*g*prob_dist-1)*self.dt) #Here we use mu=\pm1 in our units
            psi *= pot_evo
            FFT_psi = fft(psi)
            FFT_psi *= kin_evo
            psi = ifft(FFT_psi)
            norm_psi = np.trapz(np.abs(psi)**2, dx=dx)
            psi *= np.sqrt(self.Natoms/norm_psi)
            error = np.max(abs(psi_old-psi))
            n+=1
            evo_array[:,n] = psi
        evo_array = evo_array[:,0:n]
        if n>= nmax:
            print("Given tolerance not reached, simulation stopped after ", nmax, " loops")
        return evo_array

    def time_evolution(self, k_grid, psi_guess, V, t, sign = -1):
        """Uses the split step method to evolve the Gross-Pitaevskii equation over time"""
        g = self.L/self.Natoms
        i = 0+1j
        time = 0
        counter = 0
        #Initialise the kinetic evolution operator which is constant through the loop
        kin_evo = np.exp(-i/4*(k_grid**2)*self.dt)
        psi = psi_guess.astype(np.complex128)
        n_steps = int(t / self.dt)
        evo_array = np.zeros((self.n, n_steps + 2), dtype=complex)
        evo_array[:,0] = psi_guess
        #Loop over time evolution for set amount of steps
        while time<t:
            FFT_psi = fft(psi)
            FFT_psi *= kin_evo
            psi = ifft(FFT_psi)
            prob_dist = np.abs(psi)**2
            pot_evo = np.exp(-i*(V[:,counter]-sign*g*prob_dist-1)*self.dt)
            psi *= pot_evo
            FFT_psi = fft(psi)
            FFT_psi *= kin_evo
            psi = ifft(FFT_psi)
            time+=self.dt
            counter += 1
            evo_array[:,counter] = psi
        return evo_array

    #------- Visualization ----------------------------------------------------------------
    def timeslider_plot(self, x_grid, evo_array, V):
        n_pts, n_times = evo_array.shape

        # ---- Precomputation -------------------------------------------------
        # Global max density (vertical scale stays constant)
        global_ymax = np.max(np.abs(evo_array)**2)
        y_min, y_max = 0, global_ymax
        # Precompute density and wrapped phases for all timesteps
        densities = np.abs(evo_array)**2
        phases = np.angle(evo_array)
        phases = (phases + np.pi) % (2 * np.pi) - np.pi
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
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)
        # Initial density line
        line, = ax.plot(x_grid, densities[:, 0], lw=2)
        line2, = ax.plot(x_grid, V[:,0]*10, lw=2)
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
        fig.colorbar(im, ax=ax, label='Phase')
        #Create slider
        axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        time_slider = plt.Slider(ax=axtime, label='Time',
                                  valmin=0, valmax=(n_times-1)*self.dt, valinit=0)

        # ---- Update function ----------------------------------------------------
        def update(val):
            t = int(time_slider.val/self.dt)
            # Update density line
            line.set_ydata(densities[:, t])
            line2.set_ydata(V[:,t-1]*10)
            # Update polygon clip
            poly.set_xy(all_polys[t].get_xy())
            # Update phase image
            Z[:] = phases[:, t]
            im.set_data(Z)

            fig.canvas.draw_idle()

        time_slider.on_changed(update)

        plt.show()