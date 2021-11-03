import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as ticker
import sys, os, sympy, math, time
start = time.clock()

def main():
    os.mkdir("figures/")
    savedir = "figures/"
    path = os.path.join(savedir, "mesh_independent")
    
    print("Current Working Directory " , os.getcwd())
    
    influence_of_eta(savedir)
    influence_of_kappa_new(savedir)
    
    os.mkdir(path)
    savedir_mesh_indpt = "figures/mesh_independent/"
    mesh_independent_study(savedir_mesh_indpt)
     
    return

def mesh_independent_study(savedir):
    #2021 results
    ################################################################################
    # mesh independence, tol = 5e-8, AS_results_728, intial guess Sin 2...
    ################################################################################
    h_sin2_unstruct = [0.006305, 0.006012, 0.005656, 0.005248, 0.005105, 0.004791, 0.004534, 0.004326]
    obj_sin2_unstruct = [0.0207385163, 0.0208657703, 0.0207340376, 0.0207219723, 0.0207223278, 0.0206118524, 0.0206543940, 0.0205656116]
    

    h_sin2_struct = [0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
    obj_sin2_struct = [0.0207900654, 0.0206651929, 0.0205922234, 0.0205915080, 0.0206251401, 0.0204864481]
    
    fig1 = plt.figure(1)
    sin2_unstruct, = plt.plot(h_sin2_unstruct, obj_sin2_unstruct, color='black', linestyle='solid', linewidth = 2, marker='^', markerfacecolor='black', markersize=6)

    sin2_struct, = plt.plot(h_sin2_struct, obj_sin2_struct, color='red', linestyle='dashed', linewidth = 2, marker='v', markerfacecolor='red', markersize=6)

    plt.legend([sin2_unstruct, sin2_struct], ['Un-structured mesh', 'Structured mesh'], prop={'size': 7})
    # naming the x axis
    plt.xlabel('Mesh size $h$')
    # naming the y axis
    plt.ylabel('Function value')
    # giving a title to my graph
    plt.title('Mesh convergence: Initial guess with $n = 2$')
    plt.savefig(savedir + 'mesh_convergence_sin2.png')
    plt.savefig(savedir + 'mesh_convergence_sin2.pdf')


    ################################################################################
    # mesh independence, tol = 5e-8, AS_results_728, Initial guess Sin 8
    ################################################################################
    h_sin8_unstruct = [0.006305, 0.006012, 0.005656, 0.005248, 0.005105, 0.004791, 0.004534, 0.004326]
    obj_sin8_unstruct = [0.0208063357, 0.0208209401, 0.0207281684, 0.0207088930, 0.0207101645, 0.0207203492, 0.0206790310, 0.0206493411]
    
    h_sin8_struct = [0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
    obj_sin8_struct = [0.0206984970, 0.0206780927, 0.0206483581, 0.0206449207, 0.0205791574, 0.0204830673]

    fig3 = plt.figure(2)
    sin8_unstruct, = plt.plot(h_sin8_unstruct, obj_sin8_unstruct, color='black', linestyle='solid', linewidth = 2, marker='^', markerfacecolor='black', markersize=6)

    sin8_struct, = plt.plot(h_sin8_struct, obj_sin8_struct, color='red', linestyle='dashed', linewidth = 2, marker='v', markerfacecolor='red', markersize=6)

    plt.legend([sin8_unstruct, sin8_struct], ['Un-structured mesh', 'Structured mesh'], prop={'size': 7})
    # naming the x axis
    plt.xlabel('Mesh size $h$')
    # naming the y axis
    plt.ylabel('Function value')
    # giving a title to my graph
    plt.title('Mesh convergence: Initial guess with $n = 8$')
    plt.savefig(savedir + 'mesh_convergence_sin8.png')
    plt.savefig(savedir + 'mesh_convergence_sin8.pdf')

    ################################################################################
    # mesh independence, tol = 5e-8, AS_results_728, Initial guess Sin 10
    ################################################################################
    h_sin10_unstruct = [0.006305, 0.006012, 0.005656, 0.005248, 0.005105, 0.004791, 0.004534, 0.004326]
    obj_sin10_unstruct = [0.0208714772, 0.0207444765, 0.0207667310, 0.0207248011, 0.0206639308, 0.0206426684, 0.0206299958, 0.0206316756]
    
    h_sin10_struct = [0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
    obj_sin10_struct = [0.0207643837, 0.0207079952, 0.0206625309, 0.0206489540, 0.0205606049, 0.0204909709]


    fig4 = plt.figure(3)
    sin10_unstruct, = plt.plot(h_sin10_unstruct, obj_sin10_unstruct, color='black', linestyle='solid', linewidth = 2, marker='^', markerfacecolor='black', markersize=6)

    sin10_struct, = plt.plot(h_sin10_struct, obj_sin10_struct, color='red', linestyle='dashed', linewidth = 2, marker='v', markerfacecolor='red', markersize=6)

    plt.legend([sin10_unstruct, sin10_struct], ['Un-structured mesh', 'Structured mesh'], prop={'size': 7})
    # naming the x axis
    plt.xlabel('Mesh size $h$')
    # naming the y axis
    plt.ylabel('Function value')
    # giving a title to my graph
    plt.title('Mesh convergence: Initial guess with $n = 10$')
    plt.savefig(savedir + 'mesh_convergence_sin10.png')
    plt.savefig(savedir + 'mesh_convergence_sin10.pdf')

    ###############################################################
    # mesh independence, tol = 5e-8, all initial guesses
    ###############################################################


    fig6 = plt.figure(4)
    sin2_unstruct, = plt.plot(h_sin2_unstruct, obj_sin2_unstruct, color='black', linestyle='solid', linewidth = 2, marker='^', markerfacecolor='red', markersize=6)
    sin2_struct, = plt.plot(h_sin2_struct, obj_sin2_struct, color='black', linestyle='solid', linewidth = 2, marker='v', markerfacecolor='red', markersize=6)

    sin8_unstruct, = plt.plot(h_sin8_unstruct, obj_sin8_unstruct, color='green', linestyle='dashed', linewidth = 2, marker='P', markerfacecolor='yellow', markersize=6)
    sin8_struct, = plt.plot(h_sin8_struct, obj_sin8_struct, color='green', linestyle='dashed', linewidth = 2, marker='*', markerfacecolor='yellow', markersize=6)

    sin10_unstruct, = plt.plot(h_sin10_unstruct, obj_sin10_unstruct, color='violet', linestyle='-.', linewidth = 2, marker='h', markerfacecolor='blue', markersize=6)
    sin10_struct, = plt.plot(h_sin10_struct, obj_sin10_struct, color='violet', linestyle='-.', linewidth = 2, marker='H', markerfacecolor='blue', markersize=6)


    plt.legend([sin2_unstruct, sin2_struct, sin8_unstruct, sin8_struct, sin10_unstruct, sin10_struct], ['Un-structured mesh, n = 2', 'Structured mesh, n = 2', 'Un-structured mesh, n = 8', 'Structured mesh, n = 8', 'Un-structured mesh, n = 10', 'Structured mesh, n = 10',], prop={'size': 7})
    
    plt.subplots_adjust(bottom=.15, left=.15)
    # naming the x axis
    plt.xlabel('Mesh size $h$')
    # naming the y axis
    plt.ylabel('Function value')

    # giving a title to my graph
    plt.title('Mesh convergence study')
    plt.savefig(savedir + 'mesh_convergence_all_initial_guess.png')
    plt.savefig(savedir + 'mesh_convergence_all_initial_guess.pdf')

    return
    
def influence_of_eta(savedir):
    # x axis values
    eta = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1]
    # corresponding y axis values
    volume_ratio_percentage = [65.4606537484, 53.0327833000, 46.0175758157, 38.4555500405, 31.4207410805, 26.4459569991, 22.0600209868, 20.1137719708, 19.0692567687]


    # plotting the points
    plt.figure(5)
    eta_fig, = plt.plot(eta, volume_ratio_percentage, color='black', linestyle='dashed', linewidth = 2, marker='*', markerfacecolor='blue', markersize=12, label='$V/V_0$')

    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=10*len(volume_ratio_percentage)))
    legend = plt.legend(shadow=True)
    # naming the x axis
    plt.xlabel('Parameter $\eta$')
    # naming the y axis
    plt.ylabel('Volume ratio percentage')

    # giving a title to my graph
    #plt.title('The influence of the volume penalization parameter $\eta$')
    plt.savefig(savedir + "/influence_of_eta.pdf")
    plt.savefig(savedir + "/influence_of_eta.png")
    #######################################

    return
    
def influence_of_kappa_new(savedir):
    # x axis values
    kappa = [1e-15, 5e-6, 8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4]
    # corresponding y axis values
    objective_val = [0.0200480407, 0.0201142642, 0.0207649602, 0.0208034001, 0.0211766914, 0.0217101034, 0.0220751199, 0.0223785080]

    work_plus_compliance = [0.0200480407, 0.0199779737, 0.0202488222, 0.0202892008, 0.0205427093, 0.0209087532, 0.0211051777, 0.0212325515]
    
    perimeter = [0.0000000000, 0.0001362905, 0.0005161379, 0.0005141993, 0.0006339821, 0.0008013501, 0.0009699422, 0.0011459566]
    
    # plotting the points
    plt.figure(6)
    plt.figure(figsize= (6, 4))
    
    kappa_fig1, = plt.plot(kappa, objective_val, color='black', linestyle='-', linewidth = 2, marker='^', markerfacecolor='red', markersize=12, label='Function value')
    
    plt.plot(kappa, work_plus_compliance, color='black', linestyle='dashed', linewidth = 2, marker='*', markerfacecolor='green', markersize=12, label='Work plus complaince')
    
    legend = plt.legend(shadow=True)
    
    # naming the x axis
    plt.xlabel('Parameter $\kappa$')
    # naming the y axis
    plt.ylabel('Function value')
    plt.subplots_adjust(bottom=.15, left=.15)
    
    # giving a title to my graph
    #plt.title('The influence of the volume penalization parameter $\eta$')
    plt.savefig(savedir + "/influence_of_kappa_function_value.pdf")
    plt.savefig(savedir + "/influence_of_kappa_function_value.png")
    #######################################

    plt.figure(7)
    plt.figure(figsize= (6, 4))
    
    kappa_fig2, = plt.plot(kappa, perimeter, color='black', linestyle='-.', linewidth = 2, marker='d', markerfacecolor='blue', markersize=8, label='Perimeter penalization')
    
    legend = plt.legend(shadow=True)
    # naming the x axis
    plt.xlabel('Parameter $\kappa$')
    # naming the y axis
    plt.ylabel('Function value')
    plt.subplots_adjust(bottom=.15, left=0.15)
    # giving a title to my graph
    #plt.title('The influence of the volume penalization parameter $\eta$')
    plt.savefig(savedir + "/influence_of_kappa_perimeter_penalization_value.pdf")
    plt.savefig(savedir + "/influence_of_kappa_perimeter_penalization_value.png")
    
    
    return
if __name__== "__main__":
    start = time. time()
    main()
    end = time. time()
    print("\nExecution time (in seconds):",(end - start))
