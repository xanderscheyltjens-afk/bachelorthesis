from math import e
import Tijdsevolutie_Gross_Pitaevskii as GP
import numpy as np
import csv

def main():
    # --- Initialize system parameters ---
    L = 50
    n = 2**9
    dt = 0.1 * (L/n)**2
    Natoms = 10 * n

    BEC = GP.Gross_Pitaevskii_1D(L, n, dt, Natoms)
    BEC.stability_test()
    x_grid, k_grid = BEC.initialize_grids()

    psi_guess = BEC.guess_wave_function(x_grid)

    options = {
        "1":("Finite potential well"),
        "2":("Finite potential well with added Gaussian curve"),
        "3":("Harmonic potential"),
        "4":("Gravitational potential"),
        "5":("No external potential"),
        "6":("Exit program")
        }
    while True:
        print("\033[2J\033[H", end="")
        t = float(input("\nPlease choose the time over which you want to simulate:"))
        print("\033[2J\033[H", end="")
        print("\n -------Menu of possible external potentials-------")
        for key, (desc) in options.items():
            print(f"{key}. {desc}")

        choice = input("\nPlease choose which potential you want to find the ground state of: ")

        if choice == "6":
            break
            print("\033[2J\033[H", end="")
        elif choice == "1":
            V = BEC.potential_well(t=t, width = 49, height = 5)
            [evo_array_ground,k_evo_array_ground] = BEC.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-4), nmax = 10**6)
        elif choice == "2":
            V = BEC.potential_well_with_Gauss(x_grid, t)
            [evo_array_ground,k_evo_array_ground] = BEC.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-4), nmax = 10**6)
        elif choice == "3":
            V = BEC.harmonic_potential(x_grid, t, omega = 0.01)
            [evo_array_ground,k_evo_array_ground] = BEC.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-4), nmax = 10**6)
        elif choice == "4":
            V = BEC.gravity_potential(x_grid,t)
            [evo_array_ground,k_evo_array_ground] = BEC.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-4), nmax = 10**6)
        elif choice == "5":
            V = BEC.null_potential(t)
            [evo_array_ground,k_evo_array_ground] = BEC.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-4), nmax = 10**6)
        else:
            print("Invalid option, please retry")
            break
 
        ground_state = evo_array_ground[:,-1]
        print("\033[2J\033[H", end="")
        plot = input("Do you want to plot this ground state\n")
        if plot == "y" or plot == "yes":
            BEC.timeslider_plot(x_grid,evo_array_ground,V)
        options2 = {
            "1": "Static simulation of ground state",
            "2": "Added Gaussian pulse at t=1",
            "3": "Added wave pulse at t=1",
            "4": "Idealized beamsplitter", 
            "5": "Release from potential"
        }

        if choice == "1":
            options2.update({
                "6": "Added Gauss curve to potential well (same as Mathematica example)"
            })

        print("\033[2J\033[H", end="")
        print("\n -------BEC Simulation Menu-------")
        for key, desc in options2.items():
            print(f"{key}. {desc}")

        choice2 = int(input("\nChoose which simulation you'd like to run: "))
        if choice2 == 0:
            break
        elif choice2 == 1:
            [evo_array, k_evo_array] = BEC.time_evolution(k_grid,ground_state, V, t)
        elif choice2 == 2:
            V+=BEC.Gauss_pulse(x_grid, t, pulse_width_x = 20)
            [evo_array, k_evo_array] = BEC.time_evolution(k_grid,ground_state, V, t)
        elif choice2 == 3:
            V+=BEC.wave_pulse(x_grid,t)
            [evo_array, k_evo_array] = BEC.time_evolution(k_grid,ground_state, V, t)
        elif choice2 == 4:
            ground_state = BEC.split_beams(ground_state, x_grid, 0.1)
            [evo_array, k_evo_array] = BEC.time_evolution(k_grid,ground_state, V, t)
        elif choice2 == 5:
            V = BEC.null_potential(t)
            [evo_array, k_evo_array] = BEC.time_evolution(k_grid,ground_state, V, t)
        elif choice2 == 6:
            V = BEC.potential_well_with_Gauss(x_grid, t)
            [evo_array, k_evo_array] = BEC.time_evolution(k_grid,ground_state, V, t)
        else:
            print("Invalid option, please retry")
        BEC.timeslider_plot(x_grid,evo_array,V)

def quality_of_splitting():
    #Initialize system parameters
    L = 50
    n = 2**9
    dt = 0.1*(L/n)**2
    Natoms = 10*n
    t = 5
    #Initialize class
    BEC = GP.Gross_Pitaevskii_1D(L,n,dt,Natoms)
    g = np.arange(-1,1.1,0.1)
    counter = 0
    split1 = np.zeros(len(g))
    split2 = np.zeros(len(g))
    split3 = np.zeros(len(g))
    for sign in g:
        t_c = [0.5, 2.3, 4.1]
        dk_fft = 2*np.pi / L
        m = 100 
        dk = m*dk_fft
        wavelen = [2*np.pi / dk, 2*np.pi / dk, 2*np.pi / dk]
        V = 50
        omega = dk**2 / 2
        v = [omega/dk, omega/dk, omega/dk]
        q = [v[0]-0.5*dk,v[1]-0.5*dk,v[2]-0.5*dk]
        pulse_width_t = BEC.Ramsey_sequence_generator(v,q,dk,V)
        [evo_array, k_evo_array] = BEC.interferometer(t, t_c, pulse_width_t, wavelen, V, v, dk = dk, sign = sign, plot = False)

        #Calculate the fraction of the condensate that split off or is still to determine quality at 3 different points
        moment1 = 1
        percent = BEC.split_percent(k_evo_array,moment1,dk)
        split1[counter] = percent
        print("The split off part is ", percent, "% of the total condensate at t=", moment1)
        moment2 = 2.8
        percent2 = BEC.split_percent(k_evo_array,moment2,dk)
        split2[counter] = percent2
        print("The split off part is ", percent2, "% of the total condensate at t=", moment2)
        moment3 = 4.6
        percent3 = BEC.still_percent(k_evo_array,moment3)
        split3[counter] = percent3
        print("The still condensate is ", percent3, "% of the total condensate at t=", moment3)
        counter +=1

    filename = ('pulse_quality.txt')
    outfile = open(filename, 'w')
    outfile.writelines("List of percentage split off from condensate after first pulse\n:")
    outfile.writelines([str(i)+ "\n" for i in split1])
    outfile.writelines("List of percentage split off from condensate after second pulse\n:")
    outfile.writelines([str(i)+ "\n" for i in split2])
    outfile.writelines("List of recovered condensate after all pulses\n:")
    outfile.writelines([str(i)+ "\n" for i in split3])
    outfile.close()

def optimal_settings(sign, resm=10, resA=10):
    # Initialize system parameters
    L = 50
    n = 2**9
    dt = 0.1*(L/n)**2
    Natoms = 10*n
    t = 3
    BEC = GP.Gross_Pitaevskii_1D(L, n, dt, Natoms)
    # Compute the ground state beforehand for optimisation
    x_grid, k_grid = BEC.initialize_grids()
    psi_guess = BEC.guess_wave_function(x_grid)
    #V = self.gravity_potential(x_grid, t)
    V = BEC.harmonic_potential(x_grid, t, omega = 0.1, x_c = -12.5)
    [evo_array_ground, k_evo_array_ground] = BEC.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-5), nmax = 10**6, sign = sign)
    ground_state = evo_array_ground[:,-1]
    # Set the constants outside the loop
    t_c = [0.1, 0.9, 1.7]
    dk_fft = 2*np.pi / L
    #Open the CSV file we want to write the results in
    with open(f'sweep_sign{sign}_resm{resm}_resV{resA}_last_little_bit.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['m', 'V', 'split1', 'still1', 'split2', 'still2', 'still3'])

        for m in range(244, 255+resm, resm):
            for A in range(0+resA, 100+resA, resA):
                dk = m*dk_fft
                wavelen = [2*np.pi/dk, 2*np.pi/dk, 2*np.pi/dk]
                omega = dk**2 / 2
                v = [omega/dk, omega/dk, omega/dk]
                q = [v[0]-0.5*dk, v[1]-0.5*dk, v[2]-0.5*dk]
                pulse_width_t = BEC.Ramsey_sequence_generator(v, q, dk, A)
                V = BEC.potential_well(t, width = 49.9, height = 1000)
                V += BEC.wave_pulse_series(x_grid,t, t_c, pulse_width_t , wavelen, A, v)
                [evo_array, k_evo_array] = BEC.time_evolution(k_grid, ground_state, V, t, sign)

                split1 = BEC.split_percent(k_evo_array, 0.5, dk)
                still1 = BEC.still_percent(k_evo_array, 0.5, dk)
                split2 = BEC.split_percent(k_evo_array, 1.3, dk)
                still2 = BEC.still_percent(k_evo_array, 1.3, dk)
                still3 = BEC.still_percent(k_evo_array, 2.1, dk)

                writer.writerow([m, A, split1, still1, split2, still2, still3])
                print(f'm={m}, A={A} done')  # progress indicator since this will take a while
                f.flush()

def effect_of_gravity(m = 129, A=34, resg = 0.0001, limg = 2*4.61*10**(-3)):
    #Initialize system parameters
    L = 50
    n = 2**9
    dt = 0.1*(L/n)**2
    Natoms = 10*n
    t = 3
    sign = 0
    BEC = GP.Gross_Pitaevskii_1D(L,n,dt,Natoms)
    # Compute the ground state beforehand for optimisation
    x_grid, k_grid = BEC.initialize_grids()
    psi_guess = BEC.guess_wave_function(x_grid)
    V = BEC.harmonic_potential(x_grid, t, omega = 0.1, x_c = -12.5)
    [evo_array_ground, k_evo_array_ground] = BEC.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-5), nmax = 10**6, sign = sign)
    ground_state = evo_array_ground[:,-1]
    dk_fft = 2*np.pi / L
    dk = m*dk_fft
    t_c = [0.1, 0.9, 1.7]
    wavelen = [2*np.pi / dk, 2*np.pi / dk, 2*np.pi / dk]
    omega = dk**2 / 2
    v = [omega/dk, omega/dk, omega/dk]
    q = [v[0]-0.5*dk,v[1]-0.5*dk,v[2]-0.5*dk]
    pulse_width_t = BEC.Ramsey_sequence_generator(v,q,dk,A)
    moment = 2.1
    with open(f'sweep_g_resg_{resg}_limg_{limg}_m_{m}_A_{A}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['g', 'still', 'split'])
        for g in np.arange(0, limg+resg, resg):
            V = BEC.potential_well(t, width = 49.9, height = 1000)
            V += BEC.wave_pulse_series(x_grid,t, t_c, pulse_width_t , wavelen, A, v)
            V += BEC.gravity_potential(x_grid, t, rico=g)
            [evo_array, k_evo_array] = BEC.time_evolution(k_grid, ground_state, V, t, sign)
            still = BEC.still_percent(k_evo_array,moment,dk)
            split = BEC.split_percent(k_evo_array,moment,dk)
            writer.writerow([g, still, split])
            print(f'g={g} done')
            f.flush()

def standard_test():
     #Initialize system parameters
    L = 50
    n = 2**9
    dt = 0.1*(L/n)**2
    print(dt)
    Natoms = 10*n
    t = 3
    sign = 0

    #Initialize class
    BEC = GP.Gross_Pitaevskii_1D(L,n,dt,Natoms)
    #Set interferometer parameters
    t_c = [0.1, 0.9, 1.7]
    dk_fft = 2*np.pi / L
    m = 129
    #Soft lower limit. Decreasing past m=35-40 gives larger and larger knockback decreasing the quality of the interferometer
    #Upper working limit is m=255 after that it sends the condesate back??? and then some aliasing happens at much higher values
    dk = m*dk_fft
    wavelen = [2*np.pi / dk, 2*np.pi / dk, 2*np.pi / dk]
    V = 34
    #Seems like increasing this too much does do some Bragg spectroscopy since the quality decreases a lot but needs more testing
    #Decreasing too much fucks up the pulse timings since they bleed into eachother, but otherwise no worries
    omega = dk**2 / 2
    #Giving an extra offset to omega will shift q from zero. We can see at higher offsets it moves the condensate to higher harmonics (is that term correct?)
    #If used (to perhaps get to lower dk's) be sure to keep in mind the losses to higher harmonics
    v = [omega/dk, omega/dk, omega/dk]
    q = [v[0]-0.5*dk,v[1]-0.5*dk,v[2]-0.5*dk]
    #Full Ramsey sequence
    pulse_width_t = BEC.Ramsey_sequence_generator(v,q,dk,V)
    print(pulse_width_t)
    [evo_array, k_evo_array] = BEC.interferometer_in_gravity(t, t_c, pulse_width_t, wavelen, V, v, dk = dk, sign = sign, plot = True, rico = 0.1) # Accurate g = 4.61*10**(-3) for 87Ru according to course notes
    moment = 2
    percent = BEC.still_percent(k_evo_array,moment,dk)
    print("The still part is ", percent, "% of the total condensate at t=", moment)

def squeeze_k_space(minsign = -5, maxsign = 0, step = 0.1, plot = False):
    # Initialize system parameters
    L = 50
    n = 2**9
    dt = 0.1*(L/n)**2
    Natoms = 10*n
    t = 3
    BEC = GP.Gross_Pitaevskii_1D(L, n, dt, Natoms)
    x_grid, k_grid = BEC.initialize_grids()
    psi_guess = BEC.guess_wave_function(x_grid)
    # Set the constants outside the loop
    t_c = [0.1, 0.9, 1.7]
    dk_fft = 2*np.pi / L
    m = 129
    A = 34
    dk = m*dk_fft
    wavelen = [2*np.pi/dk, 2*np.pi/dk, 2*np.pi/dk]
    omega = dk**2 / 2
    v = [omega/dk, omega/dk, omega/dk]
    q = [v[0]-0.5*dk, v[1]-0.5*dk, v[2]-0.5*dk]
    pulse_width_t = BEC.Ramsey_sequence_generator(v, q, dk, A)
    signs = np.arange(minsign, maxsign+step, step)
    #Open the CSV file we want to write the results in
    with open(f'squeeze_minsign{minsign}_maxsign{maxsign}_stepsize{step}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['interaction strength','recovery'])

        for sign in signs:
            
            V = BEC.harmonic_potential(x_grid, t, omega = 0.1, x_c = -12.5)
            [evo_array_ground, k_evo_array_ground] = BEC.find_ground_state(k_grid, psi_guess, V, TOL = 10**(-5), nmax = 10**6, sign = sign)
            ground_state = evo_array_ground[:,-1]
            V = BEC.potential_well(t, width = 49.9, height = 1000)
            V += BEC.wave_pulse_series(x_grid,t, t_c, pulse_width_t , wavelen, A, v)
            [evo_array, k_evo_array] = BEC.time_evolution(k_grid, ground_state, V, t, sign = 0)
            if plot:
                BEC.timeslider_plot(x_grid, evo_array, V)
                BEC.reciprocal_timeslider_plot(k_grid,k_evo_array, dk)
            recovery = BEC.still_percent(k_evo_array, 2.1, dk)
            writer.writerow([sign, recovery])
            print(f'sign={sign} done')  # progress indicator since this will take a while
            f.flush()

def clean_up_oopsie():
    with open('sweep_sign0_resm1_resV1_2.csv', 'r') as f:
        rows = list(csv.reader(f))

        # Fix the V column (index 1) — repeating 10,20,...,100 for each m
        n_V = 100  # adjust if your resV was different
        correct_V = list(range(1, 101))

        for i, row in enumerate(rows[1:], start=1):  # skip header
            rows[i][1] = str(correct_V[(i-1) % n_V])

        with open('sweep_sign0_resm1_resV1_2_fixed.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
if __name__ == "__main__":
    #main()
    #quality_of_splitting()
    #optimal_settings(0, 1, 1)
    #effect_of_gravity()
    standard_test()
    #squeeze_k_space(minsign = -1.1, maxsign = 0.8, step = 0.1, plot = True)
 
   

    #Best simulation of all time: m=40+89, V = 34 according to matlab I think. I might need to search further in the parameter space, goddammit
    #For repelling interactions it seems like m = 40+117 and V=1