#Produces all used plots in the thesis
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import qrcode


def example():
    sns.set_theme(style="black")
    rs = np.random.RandomState(50)

    # Set up the matplotlib figure
    f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

    # Rotate the starting point around the cubehelix hue circle
    for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

        # Create a cubehelix colormap to use with kdeplot
        cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

        # Generate and plot a random bivariate dataset
        x, y = rs.normal(size=(2, 50))
        sns.kdeplot(
            x=x, y=y,
            cmap=cmap, fill=True,
            clip=(-5, 5), cut=10,
            thresh=0, levels=15,
            ax=ax,
        )
        ax.set_axis_off()

    ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
    f.subplots_adjust(0, 0, 1, 1, .08, .08)
    plt.show()

def Rabi_oscillations():
    t = np.linspace(0,2*np.pi,1000)
    curve1 = np.cos(t/2)**2
    curve2 = np.sin(t/2)**2
    lines = [(t, curve1), (t, curve2)]
    fig, ax = plt.subplots()
    ax.plot(t,curve1,label=r"$|1\rangle$")
    ax.plot(t,curve2,"-.", color="k", label=r"$|2\rangle$")
    ax.grid()
    ax.set_xlabel("$\Omega t$")
    ax.set_xlim(0,2*np.pi)
    ax.set_ylim(0,1)
    ax.set_ylabel("Kans $|c_i|^2$")
    plt.title(r"Rabi oscillaties op resonantie $\omega=\omega_{12}$")
    ax.legend()

def datareader(name):
    path = "C:\\Users\\xande\\OneDrive\\Bureaublad\\Bachelor fysica\\Ba 3\\Bachelorproef\\Code\\Tijdsevolutie_Gross-Pitaevskii\\Data"
    with open(path+f"\\{name}.csv", 'r') as f:
        file = csv.reader(f)
        header = next(file)  
        data = list(file)
    return data

def parameter_space_plot(name):
    data = datareader(name)

    m_vals = np.array([float(row[0]) for row in data])
    A_vals = np.array([float(row[1]) for row in data])
    still3 = np.array([float(row[7]) for row in data])

    m_unique = np.unique(m_vals)
    A_unique = np.unique(A_vals)
    n_m = len(m_unique)
    n_A = len(A_unique)

    still3_grid = still3.reshape(n_m, n_A)
    M, A = np.meshgrid(m_unique, A_unique)
    M, A = np.transpose(M), np.transpose(A)

    sns.set_theme(style="dark")
    #cmap = sns.cubehelix_palette(n_colors= 10, light=1, dark=0, as_cmap=True)

    fig, ax = plt.subplots()
    cf = ax.contourf(M, A, still3_grid, cmap="plasma", levels=30)
    ax.set_ylabel("Amplitude (A)")
    ax.set_xlabel("Impulsstoot ($k_{kick}/dk$)")
    fig.colorbar(cf, ax=ax)
    return m_unique,A_unique, ax

def RWA_valid(name, sim_length, gamma=1):
    m_unique, A_unique, ax = parameter_space_plot(name)
    dk = 2*np.pi/sim_length
    A_threshold = ((m_unique*dk/2+1)*m_unique*dk)/gamma
    ax.plot(m_unique, A_threshold, lw=2, color='k')
    ax.set_ylim(1,np.max(A_unique))
    ax.set_xlim(1,np.max(m_unique))

def effect_of_gravity_plot(name):
    data = datareader(name)
    g_vals = np.array([float(row[0]) for row in data])
    still = np.array([float(row[1]) for row in data])
    split = np.array([float(row[2]) for row in data])

    #Renormalize
    still = still/still[0]
    split = split/split[0]

    fig, ax = plt.subplots()
    ax.plot(g_vals, still, ".", label="Simulatie")
    ax.legend()
    ax.grid()
    ax.set_xlim(min(g_vals), max(g_vals))
    ax.set_xlabel("Valversnelling")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fractie teruggewonnen atomen")
    return ax, g_vals

def make_qr(link, filename="qr_code.png", size=10, border=4):
    """
    Generate a QR code from a link and save it as an image.

    Parameters:
    - link (str): URL to encode
    - filename (str): output file name
    - size (int): controls overall size (bigger = higher resolution)
    - border (int): white border thickness (important for scanning)
    """
    qr = qrcode.QRCode(
        version=None,  # automatic size
        error_correction=qrcode.constants.ERROR_CORRECT_Q,  # good for posters
        box_size=size,
        border=border,
    )

    qr.add_data(link)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)

    print(f"QR code saved as {filename}")
if __name__=="__main__":
    #Figure to illustrate Rabi oscillations
    Rabi_oscillations()
    plt.show()
    #parameter space no interactions
    parameter_space_plot("sweep_gfactor0_resm1_resA1_startm10")
    plt.title("Parameterruimte voor BEC zonder interatomaire interacties")
    plt.show()
    #parameter space for finer grid
    parameter_space_plot("Parameter_sweep_fijn_grid")
    plt.title("Parameterruimte voor simulatie met 1024 roosterpunten")
    plt.show()
    #RWA threshold prediction gamma=2 bigger parameter space
    RWA_valid("Parameter_sweep_fijn_grid", 50, 2)
    plt.title("Parameterruimte voor simulatie met 1024 roosterpunten")
    plt.show()
    #parameter space repulsive interactions
    parameter_space_plot("sweep_gfactor-1_resm1_resA1_startm10")
    plt.title("Parameterruimte voor BEC met repulsieve interatomaire interacties")
    plt.show()
    #RWA threshold prediction gamma=1
    RWA_valid("sweep_gfactor0_resm1_resA1_startm10", 50, 1)
    plt.title(f"Drempelwaarde amplitude geplot over parameterruimte voor $\Gamma=1$")
    plt.show()
    #RWA threshold prediction gamma=2
    RWA_valid("sweep_gfactor0_resm1_resA1_startm10", 50, 2)
    plt.title(f"Drempelwaarde amplitude geplot over parameterruimte voor $\Gamma=2$")
    plt.show()
    # example()
    ax1, g_vals = effect_of_gravity_plot("gravitym129A34_T_0.8_lim_1")
    #Calculate theoretical results
    phase_diff = g_vals*(129*2*np.pi/50)*0.8**2
    still_theory = np.cos(phase_diff/2)**2
    ax1.plot(g_vals, still_theory, "-.", color="k", label='Theorie')
    ax1.legend()
    plt.title("Invloed zwaartekracht op recovery voor T=0,8")
    plt.show()
    ax2, g_vals2 = effect_of_gravity_plot("gravitym129A34_T_1.6_lim_1")
    #Calculate theoretical results
    phase_diff2 = g_vals*(129*2*np.pi/50)*1.6**2
    still_theory2 = np.cos(phase_diff2/2)**2
    ax2.plot(g_vals, still_theory2, "--", color="k", label='Theorie')
    plt.title("Invloed zwaartekracht op recovery voor T=1,6")
    ax2.legend()
    plt.show()
    #Make QR-code for the playlist
    #make_qr("https://www.youtube.com/playlist?list=PLmDD9LDO3UAvn8hoFI1lrV5Osvf7Tj9q9", "QR_code_playlist.png")
    