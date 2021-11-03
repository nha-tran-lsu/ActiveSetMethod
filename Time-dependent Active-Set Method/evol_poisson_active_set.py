from dolfin import *
from collections import OrderedDict
import numpy as np
import time, logging
from petsc4py import PETSc
import matplotlib.pyplot as plt
import sympy as sym
#import nb

start = time.clock()

set_log_level(1000)
try:
    from petsc4py import PETSc
except:
    print ("\n*** Warning: you need to have petsc4py installed for this module to run\n")
    print ("\nExiting.\n")
    exit()

if not has_petsc4py():
    print ("\n*** Warning: Dolfin is not compiled with petsc4py support\n")
    print ("\nExiting.\n")
    exit()
    
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
#set_log_active(1000)

def main():
    # Optimization options for the form compiler
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"
    
    mpi_comm = MPI.comm_world; my_rank = MPI.rank(mpi_comm)
    mpi_size = MPI.size(mpi_comm);
    
    # Pick an example:
    # 1. 1D problem clamped 2 sides
    para = default_parameters(); # load the default parameters
    para.parse(); # take inputs from the command line
    example = para["ex"]
    
    # Get the parameteres information for the_example
    [gamma, rho] = get_const_info(para)
    [case, bc_type, Lx, Ly, cell_size] = get_setting_info(para)
    [is_constant, slope, noise_level, from_manufactured_sol]= get_initial_guess_info(para)
    
    # time distiziation
    if example == 2:
        print("Reset the time discritization")
        para["time_app"]["T"] = 0.15
        para["time_app"]["nt"] = 3
        
    [nt, dt, T] = get_time_partition_info(para)
    print("The time discritization:\n T = {:.4f}; dt = {:.4f}; nt = {:d} ".format(T, dt, nt))

    # create mesh and define function spaces
    Nx = int(Lx/cell_size); Ny = int(Ly/cell_size);
    savedir = "evol_output/example_%s_case_%s"%(example, case)
    
    [mesh, boundary_markers, sub_domains, n, ds] = create_mesh(para, savedir);
    File(savedir + "/parameters_ex%s_%s.xml"%(example, case)) << para
    the_domain = [mesh, boundary_markers, sub_domains, n, ds];
    meshname = savedir + "/mesh/mesh%s%s.xdmf"%(example, case)
    XDMFFile(MPI.comm_world, meshname).write(mesh)

    
    V = FunctionSpace(mesh, 'Lagrange', 2)
    lam_D = Expression('0', degree=0, t=0)

    #Manufacturing displacement field
    if example == 2:
        [u_D, data, ctrls_data] = manufacturing_target_displacement_ex2(V, the_domain, savedir, para)
    elif example == 3:
        [u_D, data, ctrls_data] = manufacturing_target_displacement_ex3(V, the_domain, savedir, para)
    elif example == 4:
        [u_D, data, ctrls_data] = manufacturing_target_displacement_ex4(V, the_domain, savedir, para)
    else:
        [u_D, data, ctrls_data] = manufacturing_target_displacement_ex1(V, the_domain, savedir, para)
    
    # plot maximum values of the prescribed displacement
    plot_evol_max_values(data, nt, dt, savedir, example, case, "data")
    
    #create variables
    adjoints = create_variable(V, "adjoints", dt, T)
    ctrls = create_variable(V, "ctrls", dt, T)
    gradients = create_variable(V, "gradients", dt, T)
    states = create_variable(V, "states", dt, T)
    active_set_lb = create_variable(V, "active_set_lb", dt, T)
    active_set_ub = create_variable(V, "active_set_ub", dt, T)

    #Initial guess for the control variables
    initial_guess(ctrls, V, the_domain, savedir, para)
        
    # save initial control guess
    initial_ctrls_name = "ini_ctrls"
    save_seq(ctrls, initial_ctrls_name, nt, dt, savedir)
    plot_evol_max_values(ctrls, nt, dt, savedir, example, case, "ini_ctrl_guess")
    
    # the corresponding displacement values of the initial guess
    ini_displacement_solver(states, u_D, ctrls, savedir, the_domain, para)
    plot_evol_max_values(states, nt, dt, savedir, example, case, "ini_dis_guess")

    L_array = []
    #------------------------------------------------
    # Gradient Descent Algorithm
    #------------------------------------------------
    [iter_max, save_frequency, tol, step_size, step_size_min, step_size_max, search_scheme] = get_GDA_info(para)
    
    iter = 0

    fun_val_his = np.zeros((iter_max + 1, 5))
    # initialize iter counters
    converged = False

    print( "\niter:\t cost:L;\t grad_norm;\t step_size; line search flag")
    while iter < iter_max and not converged:
        #Solve the forward equations
        evol_displacement_solver(states, u_D, ctrls, savedir, the_domain, para)
        
        # plot maximum displacement initial step
        if iter == 0:
            plot_evol_max_values(states, nt, dt, savedir, example, case, "initial guess")
            
        # Compute the adjoint equations, so that we can find the current gradient
        evol_adjoint_solver_forward(adjoints, lam_D, states, data, savedir, the_domain, para)
        
        # evaluate the current gradient
        gradient_sequence_solver(gradients, adjoints, ctrls, the_domain, para)
        
        #with the current gradient, we can determine the active-set
        get_evol_active_set(states, gradients, active_set_lb, active_set_ub, para)
        
        #CORRECTION STEP: Recompute the displacement field associated to the active-set
        evol_displacement_solver_with_AS(states, u_D, ctrls, active_set_lb, active_set_ub, savedir, the_domain, para)
        
        #CORRECTION STEP: Recompute the adjoint variables associated to the displacements on the Active-Set
        evol_adjoint_solver_with_AS(adjoints, active_set_lb, active_set_ub, lam_D, states, data, savedir, the_domain, para)
        
        # Re-evaluate the gradient with the current Active-Set
        gradient_sequence_solver(gradients, adjoints, ctrls, the_domain, para)
        grad_norm = eval_total_gradient_norm(gradients, nt, dt)

        # evaluate the objective function
        L = eval_objective_function(states, data, ctrls, para)
        
        if iter == 0:
            grad_norm0 = grad_norm
         
        L_array.append(L)
        # do the line search
        [step_size, flag] = line_search(L, states, data, ctrls, gradients, grad_norm, u_D, active_set_lb, active_set_ub, savedir, the_domain, para)

        #print( "{0:5d}: {:.8e}; {:.8e}; {:.3f}".format(iter, L, grad_norm, step_size))
        print( "%3d: %8.10e; %8.10e; %2.1e; %r" % (iter, L, grad_norm, step_size, flag))
        
        #update control variables
        update_control_var(ctrls, gradients, step_size, nt, dt)
        
        fun_val_his[iter] = np.array([iter, L, grad_norm, step_size, flag])
        # save info to text file
        #-----------------------------------
        with open(savedir + '/fun_val_his_case_%s.txt'%(case),'wb') as f:
            np.savetxt(f, fun_val_his, fmt='%e')
            
        # check for convergence
        if grad_norm < tol*grad_norm0 and iter > 1:
            converged = True
            print( "Gradient Descent Algorithm converged in ",iter,"  iterations")
        if (iter% save_frequency) == 0:
            ctrls_name = "%sctrls"%iter
            save_seq(ctrls, ctrls_name, nt, dt, savedir)

            states_name = "%sstates"%iter
            save_seq(states, states_name, nt, dt, savedir)

            adjoints_name = "%sadjoints"%iter
            save_seq(adjoints, adjoints_name, nt, dt, savedir)
            
            active_set_lb_name = "%sactive_set_lb"%iter
            save_seq(active_set_lb, active_set_lb_name, nt, dt, savedir)
            
            active_set_ub_name = "%sactive_set_ub"%iter
            save_seq(active_set_ub, active_set_ub_name, nt, dt, savedir)
            
            plot_evol_max_values(states, nt, dt, savedir, example, case, "dis_at_iter%s"%iter)
            plot_evol_max_values(ctrls, nt, dt, savedir, example, case, "ctrls_at_iter%s"%iter)
            plot_loglog_objective_difference_history(iter, fun_val_his, savedir, example, case, "loglog_obj_diff_val%s"%iter)
            plot_objective_function_history(iter, fun_val_his, savedir, example, case, "obj_val%s"%iter)
            plot_gradient_norm_history(iter, fun_val_his, savedir, example, case, "grad_norm%s"%iter)
            
        iter += 1
        
    if not converged:
        print( "Gradient Descent Algorithm did not converge in ", iter_max, " iterations")
    
    line_search_checker = all(fun_val_his[:iter, 4])
    print("Is there any line search fail %r"%(not line_search_checker))
    
    
    ctrls_norm = sequence_norm(ctrls, nt, dt)
    print("ctrls_norm = ", ctrls_norm)

    #save variables
    ctrls_name = "opt_ctrls"
    save_seq(ctrls, ctrls_name, nt, dt, savedir)

    states_name = "opt_states"
    save_seq(states, states_name, nt, dt, savedir)

    adjoints_name = "opt_adjoints"
    save_seq(adjoints, adjoints_name, nt, dt, savedir)
    
    active_set_lb_name = "%sopt_active_set_lb"%iter
    save_seq(active_set_lb, active_set_lb_name, nt, dt, savedir)
    
    active_set_ub_name = "%sopt_active_set_ub"%iter
    save_seq(active_set_ub, active_set_ub_name, nt, dt, savedir)

    plot_loglog_objective_difference_history(iter, fun_val_his, savedir, example, case, "loglog_obj_val_opt%s"%iter)
    plot_objective_function_history(iter, fun_val_his, savedir, example, case, "obj_val%s_opt"%iter)
    plot_gradient_norm_history(iter, fun_val_his, savedir, example, case, "grad_norm%s_opt"%iter)
    plot_evol_max_values(states, nt, dt, savedir, example, case, "opt_result")
    
    error_2norm, error_max = compare_sequences(ctrls, ctrls_data, nt, dt)
    print("\nCompare the error of the controls")
    print("error_2norm = ", error_2norm)
    print("error_max = ", error_max)

    error_2norm, error_max = compare_sequences(states, data, nt, dt)
    print("\nCompare the error of the states")
    print("error_2norm = ", error_2norm)
    print("error_max = ", error_max)
    
    return

def get_current_pde_bounds(Vu, u_min, u_max, active_set_lb_t, active_set_ub_t):
    # get u_min arrays
    u_min_array = u_min.vector().get_local();
    u_max_array = u_max.vector().get_local();
    LAS = active_set_lb_t.vector().get_local();
    UAS = active_set_ub_t.vector().get_local();
    
    # at the step k of Primal- Dual AS Algorithm
    ui_min = interpolate(Constant(-1e6), Vu);
    ui_max = interpolate(Constant(1e6), Vu);
    
    # get arrays
    ui_max_array = ui_max.vector().get_local();
    ui_min_array = ui_min.vector().get_local();
    
    #Set upper and lower bounds for adjoint
    for at_dof, is_active_point in enumerate(LAS):
        if is_active_point == 1 :
            ui_max_array[at_dof] = u_min_array[at_dof];
            ui_min_array[at_dof] = u_min_array[at_dof];
            
    for at_dof, is_active_point in enumerate(UAS):
        if is_active_point == 1 :
            ui_max_array[at_dof] = u_max_array[at_dof];
            ui_min_array[at_dof] = u_max_array[at_dof];
        
    ui_max.vector()[:] = ui_max_array;
    ui_min.vector()[:] = ui_min_array;
    return [ui_min, ui_max]


def plot_evol_max_values(states, nt, dt, savedir, example, case, fig_name):
    
    u_evol_max = []
    ti = []
    i = 0;
    while i <= nt:
        t = float(i*dt)
        state_max = np.max(states[t].vector()[:])
        u_evol_max.append(state_max)
        ti.append(t)
        i+=1
        
    plt.clf()
    fig = plt.figure()
    
    p1,=plt.plot(ti, u_evol_max, color='k', linestyle='-', linewidth=1, marker='*', markerfacecolor='red', markersize=3,)
    plt.legend([p1], ["maximum values of %s"%fig_name])
    
    plt.xlabel('time')
    plt.ylabel('values')
    fig.savefig(savedir + "/ex%s_%s_maximum_values_%s.png"%(example, case, fig_name))

    return
    
    
def plot_loglog_objective_difference_history(iter, fun_val_his, save_dir, example, case, fig_name):
    # ---------------------------------------------------
    # Plot the objective function in loglog scale
    # ---------------------------------------------------
    plt.clf()
    fig = plt.figure()
    
    error_array = fun_val_his[:iter, 1] - fun_val_his[iter, 1]
    iter_array  = fun_val_his[:iter, 0]
    
    p1,= plt.loglog(iter_array, error_array)
    plt.legend([p1], ["difference"])
    plt.xlabel('Log(iter)')
    plt.ylabel('values')
    fig.savefig(save_dir + "/ex%s_%s_objective_values_loglog_by_%s.png"%(example, case, fig_name))
    return

def plot_objective_function_history(iter, fun_val_his, save_dir, example, case, fig_name):
    # ---------------------------------------------------
    # Plot the objective function
    # ---------------------------------------------------
    plt.clf()
    fig1 = plt.figure(1);
    fun_val_array = fun_val_his[:iter, 1]
    iter_array = fun_val_his[:iter, 0]
    p1,= plt.plot(iter_array, fun_val_array, color='k', linestyle='-.', linewidth=1, marker='*', markerfacecolor='red', markersize=2,)
    plt.legend([p1], ["cost"])
    plt.xlabel('iterations')
    plt.ylabel('cost')
    fig1.savefig(save_dir + "/ex%s_%s_objective_values_by_%s.png"%(example, case, fig_name))

    return


def plot_gradient_norm_history(iter, fun_val_his, save_dir, example, case, fig_name):
    
    # ---------------------------------------------------
    # Plot the norm of gradient function
    # ---------------------------------------------------
    plt.clf()
    fig2 = plt.figure(2);
    p1,=plt.plot(fun_val_his[:iter,0], fun_val_his[:iter,2], color='k', linestyle='-.', linewidth=1, marker='*', markerfacecolor='red', markersize=2,)
    plt.legend([p1], ["gradient norm"])
    plt.xlabel('iterations')
    plt.ylabel('values')
    fig2.savefig(save_dir + "/ex%s_%s_gradient_norm_by_%s.png"%(example, case, fig_name))
    return

def sequence_norm(seq, nt, dt):
    """Print norm of all element of a given sequence, seq, for all time step
        """
    
    seq_norm = []
    i = 0
    while i <= nt:
        t = float(i*dt)
        seq_norm.append([t, norm(seq[t])])
        i+=1

    return seq_norm

def update_control_var(ctrls, gradients, step_size, nt, dt):
    """Update the control varialbes
        ctrls = ctrls + step_size*ctrls
        """
    # at t = 0, f is given, so do not update. Let it be the initial guess
    i = 0
    
    i = 1
    while i <= nt:
        t = float(i*dt)
        ctrls[t].vector().axpy(-step_size, gradients[t].vector() )

        i+=1
    
    return

def line_search(cost_in, states, data, ctrls, gradients, grad_norm, u_D, active_set_lb, active_set_ub, savedir, the_domain, para):

    [gamma, rho] = get_const_info(para)
    
    # time distiziation
    [nt, dt, T] = get_time_partition_info(para)
    
    V = states[0].function_space()
    
    u_next = create_variable(V, "u_next", dt, T)
    f_next = create_variable(V, "f_next", dt, T)
    
    #GDA info
    [iter_max, save_frequency, tol, step_size, step_size_min, step_size_max, search_scheme] = get_GDA_info(para)
    
    if search_scheme == "constant_step_size":
        step_size = step_size
        flag = True
        out_put = [step_size, flag]
    else:
        # armijo line search
        count_step = 0; eps = 1e-5; armijo = 1e3;
        random_step_size = step_size;
        step_size = step_size_max;

        while (armijo > -eps*step_size*(grad_norm**2)) and (step_size > step_size_min):
            step_size = (0.5**count_step)*step_size_max;
        
            #Update next trial control varialbes
            copy_seq(f_next, ctrls, nt, dt)
            update_control_var(f_next, gradients, step_size, nt, dt)
        
            # solve the state/forward problem
            evol_displacement_solver_with_AS(u_next, u_D, f_next, active_set_lb, active_set_ub, savedir, the_domain, para)
        
            # evaluate objective value function
            cost_next = eval_objective_function(u_next, data, f_next, para)

            armijo = cost_next - cost_in;
        
            #print("armijo = {:.5e}; cost_next = {:.5e}; cost_in = {:.5e}".format(armijo, cost_next, cost_in))
            count_step+=1;

        # check if Armijo conditions are satisfied
        if armijo < -eps*step_size*(grad_norm**2):
            backtrack_converged = True
        else:
            backtrack_converged = False
            print( "Backtracking failed. A sufficient descent direction was not found")
            step_size = random_step_size; # choose a random step_size
        flag = backtrack_converged
        out_put = [step_size, flag]
    return out_put


def eval_total_gradient_norm(SSn, nt, dt):
    """Evaluate the norm of the gradient of the objective function over time
    """
    Vg = SSn[0].function_space()
    
    SSn_total_norm = 0.
    i = 0;
    while i <= nt:
        t = float(i*dt)
        
        if (i == 0) or (i == nt):
            weight = 0.5
        else:
            weight = 1.
        SSn_total_norm += weight*dt*norm(SSn[t])
        i+=1
    return SSn_total_norm
    

def eval_objective_function(states, data, ctrls, para):
    # The objective value function, which has the same value as Lagrange function. The difference betwen these two value is up to the error of the PDE constraint solvers.
    [gamma, rho] = get_const_info(para)
    
    # time distiziation
    [nt, dt, T] = get_time_partition_info(para)
    V = states[0].function_space()
    
    val = 0.
    i = 0;
    while i <= nt:
        t = float(i*dt)
            
        if (i == 0) or (i == nt):
            weight = 0.5
        else:
            weight = 1.
        
        val += 0.5*gamma*weight*dt*assemble((states[t] - data[t])**2*dx) + 0.5*rho*weight*dt*assemble((ctrls[t])**2*dx)
        i+=1
        
    L_val = float(val)
    return L_val
    

def initial_guess(ctrls, V, the_domain, savedir, para):
    # Get the parameteres information for the_example
    [gamma, rho] = get_const_info(para)
    [case, bc_type, Lx, Ly, cell_size] = get_setting_info(para)
    [is_constant, slope, noise_level, from_manufactured_sol]= get_initial_guess_info(para)
    # time distiziation
    [nt, dt, T] = get_time_partition_info(para)
    example = para["ex"]
    
    if example == 1:
        f = Expression('k*t', degree = 1, k = 1, t = 0)
    else:
        if is_constant == 1:
            f = Expression('k*t/T', degree=1, k = 0, t=0, T = T)
        else:
            f = Expression('k*sin(t*pi/2) + 0.01*sin(m*pi*x[0]/Lx)*sin(n*pi*x[1]/Ly)', degree = 3, Lx = Lx, Ly = Ly, m = 10, n = 10, k = 200, t = 0)    
    i = 0;
    while i <= nt:
        t = float(i*dt)
        f.k = slope; f.t = t;
        ctrls[t] = interpolate(f, V)
        i+=1
    return
    
    
def manufacturing_target_displacement_ex1(V, the_domain, savedir, para):
    # Get the parameteres information for the_example
    [gamma, rho] = get_const_info(para)
    [case, bc_type, Lx, Ly, cell_size] = get_setting_info(para)
    
    # time distiziation
    [nt, dt, T] = get_time_partition_info(para)
    
    # Manufaturing a solution: Start
    #----------------------------------------
    x, y, t = sym.symbols('x[0], x[1], t')
    u1_sym = x*(1-x)*(100*t)
    u2_sym = x*(1-x)*(100*(T - t))
    
    f1_sym = -(sym.diff(sym.diff(u1_sym, x), x) + sym.diff(sym.diff(u1_sym, y), y))
    f2_sym = -(sym.diff(sym.diff(u2_sym, x), x) + sym.diff(sym.diff(u2_sym, y), y))
    
    u1_sym = sym.simplify(u1_sym)
    u2_sym = sym.simplify(u2_sym)
    f1_sym = sym.simplify(f1_sym)
    f2_sym = sym.simplify(f2_sym)
    
    u1_code = sym.printing.ccode(u1_sym)
    u2_code = sym.printing.ccode(u2_sym)
    f1_code = sym.printing.ccode(f1_sym)
    f2_code = sym.printing.ccode(f2_sym)

    print("\n The manufactured functions are ")
    print('f1 =', f1_code)
    print('f2 =', f2_code)
    print('u1 =', u1_code)
    print('u2 =', u2_code)
    
    f1 = Expression(f1_code, degree = 3, t = 0)
    f2 = Expression(f2_code, degree = 3, t = 0)
    u1 = Expression(u1_code, degree = 3, t = 0)
    u2 = Expression(u2_code, degree = 3, t = 0)

    states_data = OrderedDict()
    ctrls_data = OrderedDict()
    
    i = 0;
    while i <= nt:
        t = float(i*dt)
        states_data[t] = Function(V, annotate = True)
        ctrls_data[t] = Function(V, annotate = True)

        if i <= (nt/2):
            u1.t = t
            states_data[t] = interpolate(u1, V)
            f1.t=t
            ctrls_data[t] = interpolate(f1, V)
        else:
            u2.t = t
            states_data[t] = interpolate(u2, V)
            f2.t=t
            ctrls_data[t] = interpolate(f2, V)
        
        i+=1
        
    states_data_name = "states_data"
    save_seq(states_data, states_data_name, nt, dt, savedir)
    
    ctrls_data_name = "ctrls_data"
    save_seq(ctrls_data, ctrls_data_name, nt, dt, savedir)

    #since u1 = u2 on the boundary, we can pick either of them
    variables = [u1, states_data, ctrls_data]
    return variables


def manufacturing_target_displacement_ex4(V, the_domain, savedir, para):
    # Get the parameteres information for the_example
    [gamma, rho] = get_const_info(para)
    [case, bc_type, Lx, Ly, cell_size] = get_setting_info(para)
    # time distiziation
    [nt, dt, T] = get_time_partition_info(para)
    
#    f = Expression('val*sin(t*pi)*exp(-100*((x[0]- sin(t*pi/3))*(x[0]- sin(t*pi/3)) + (x[1]- 0.5)*(x[1]- 0.5)))', degree = 3, val = 100, t = 0)
    f = Expression('val*exp(-100*((x[0]- sin(t*pi/3))*(x[0]- sin(t*pi/3)) + (x[1]- 0.5)*(x[1]- 0.5)))', degree = 3, val = 200, t = 0)
    
    states_data = OrderedDict()
    ctrls_data = OrderedDict()
    
    u_min = interpolate(Constant(-1e6), V);
    u_max = interpolate(Constant(1e6), V);
    u_D = Expression('0', degree=0, t=0)
    
    i = 0;
    while i <= nt:
        t = float(i*dt)
        states_data[t] = Function(V, annotate = True)
        ctrls_data[t] = Function(V, annotate = True)
        f.val = 200; f.t = t;
        
        if i == 0:
            ctrls_data[t] = interpolate(Constant(0), V)
        else:
            ctrls_data[t] = interpolate(f, V)
        
        # Solve associated displacement equation
        u = displacement_solver(V, u_D, u_min, u_max, ctrls_data[t], the_domain, para)
        
        # update the state varialbes
        states_data[t].assign(u)
        i+=1
        
    states_data_name = "states_data"
    save_seq(states_data, states_data_name, nt, dt, savedir)
    
    ctrls_data_name = "ctrls_data"
    save_seq(ctrls_data, ctrls_data_name, nt, dt, savedir)

    variables = [u_D, states_data, ctrls_data]
    return variables


    
def manufacturing_target_displacement_ex3(V, the_domain, savedir, para):
    # Get the parameteres information for the_example
    [gamma, rho] = get_const_info(para)
    [case, bc_type, Lx, Ly, cell_size] = get_setting_info(para)
    # time distiziation
    [nt, dt, T] = get_time_partition_info(para)
    
    f = Expression('val*exp(-100*((x[0]- sin(t*pi/3))*(x[0]- sin(t*pi/3)) + (x[1]- 0.25)*(x[1]- 0.25)))', degree = 3, val = 200, t = 0)
    
    states_data = OrderedDict()
    ctrls_data = OrderedDict()
    
    u_min = interpolate(Constant(-1e6), V);
    u_max = interpolate(Constant(1e6), V);
    u_D = Expression('0', degree=0, t=0)
    
    i = 0;
    while i <= nt:
        t = float(i*dt)
        states_data[t] = Function(V, annotate = True)
        ctrls_data[t] = Function(V, annotate = True)
        f.val = 200; f.t = t;
        
        if i == 0:
            ctrls_data[t] = interpolate(Constant(0), V)
        else:
            ctrls_data[t] = interpolate(f, V)
        
        # Solve associated displacement equation
        u = displacement_solver(V, u_D, u_min, u_max, ctrls_data[t], the_domain, para)
        
        # update the state varialbes
        states_data[t].assign(u)
        
        i+=1
        
    states_data_name = "states_data"
    save_seq(states_data, states_data_name, nt, dt, savedir)
    
    ctrls_data_name = "ctrls_data"
    save_seq(ctrls_data, ctrls_data_name, nt, dt, savedir)

    variables = [u_D, states_data, ctrls_data]
    return variables

def manufacturing_target_displacement_ex2(V, the_domain, savedir, para):
    # Get the parameteres information for the_example
    [gamma, rho] = get_const_info(para)
    [case, bc_type, Lx, Ly, cell_size] = get_setting_info(para)
    # time distiziation
    [nt, dt, T] = get_time_partition_info(para)
    
    f1 = Expression('val*exp(-100*((x[0]- 0.25)*(x[0]- 0.25) + (x[1]- 0.25)*(x[1]- 0.25)))', degree = 3, val = 200)
    f2 = Expression('val*exp(-100*((x[0]- 0.75)*(x[0]- 0.75) + (x[1]- 0.25)*(x[1]- 0.25)))', degree = 3, val = 200)
    f3 = Expression('val*exp(-100*((x[0]- 0.25)*(x[0]- 0.25) + (x[1]- 0.25)*(x[1]- 0.25))) + val*exp(-100*((x[0]- 0.75)*(x[0]- 0.75) + (x[1]- 0.25)*(x[1]- 0.25)))', degree = 3, val = 200)
    
    states_data = OrderedDict()
    ctrls_data = OrderedDict()
    
    u_min = interpolate(Constant(-1e6), V);
    u_max = interpolate(Constant(1e6), V);
    u_D = Expression('0', degree=0, t=0)
    
    i = 0;
    while i <= nt:
        t = float(i*dt)
        states_data[t] = Function(V, annotate = True)
        ctrls_data[t] = Function(V, annotate = True)
        
        if i == 0:
            ctrls_data[t] = interpolate(Constant(0), V)
        elif i == 1:
            ctrls_data[t] = interpolate(f1, V)
        elif i == 2:
            ctrls_data[t] = interpolate(f2, V)
        else:
            ctrls_data[t] = interpolate(f3, V)
        
        # Solve associated displacement equation
        u = displacement_solver(V, u_D, u_min, u_max, ctrls_data[t], the_domain, para)
        
        # update the state varialbes
        states_data[t].assign(u)
        
        i+=1
        
    states_data_name = "states_data"
    save_seq(states_data, states_data_name, nt, dt, savedir)
    
    ctrls_data_name = "ctrls_data"
    save_seq(ctrls_data, ctrls_data_name, nt, dt, savedir)

    variables = [u_D, states_data, ctrls_data]
    return variables
    
def copy_seq(seq1, seq2, nt, dt):
    V = seq1[0].function_space()

    i = 0
    while i <= nt:
        t = float(i*dt)
        seq1[t].vector()[:] = seq2[t].vector()[:]
        i+=1

    return
    
def compare_sequences(seq1, seq2, nt, dt):
    V = seq1[0].function_space()
    
    error_2norm = np.zeros(nt+1)
    error_max = np.zeros(nt+1)
    
    i = 0
    while i <= nt:
        t = float(i*dt)
        e = Function(V)
        e.assign(seq1[t] - seq2[t])
        error_2norm[i] = np.linalg.norm(e.vector()[:])
        error_max[i] = np.abs(seq1[t].vector()[:] - seq2[t].vector()[:]).max()
        i+=1

    return error_2norm, error_max

def save_seq(seq, seq_name, nt, dt, savedir):
    
    i = 0
    file_seq = File(savedir+'/%s/%s.pvd'% (seq_name, seq_name))
    while i <= nt:
        t = float(i*dt)
        file_seq << (seq[t], t)
        i+=1
    
    return

def create_variable(V, var_name, dt, T):
    """ Create problem variables
        """
    nt = int(T/dt);

    var = OrderedDict()
    
    # define function for state and adjoint
    i = 0;
    while i <= nt:
        t = float(i*dt)
        var[t] = Function(V, name=var_name, annotate=True)
        i+= 1;

    return var
    
    
def gradient_sequence_solver(gradients, adjoints, ctrls, the_domain, para):
    """ Determine the gradient of the objective function at each time step
        """
    [gamma, rho] = get_const_info(para)
    # time distiziation
    [nt, dt, T] = get_time_partition_info(para)
    Vg = gradients[0].function_space()
    i = 0;
    while i <= nt:
        t = float(i*dt)
        if (i == 0) or (i == nt):
            weight = 0.5
        else:
            weight = 1.
        rhs = weight*(rho*ctrls[t] + adjoints[t])
        g_t = gradient_solver(Vg, rhs, the_domain)
        gradients[t].assign(g_t)
        i+=1;
    return

def gradient_solver(Vg, rhs, the_domain):
    # boundary conditions
    bcs_g = []
    # define Trial and Test Functions
    g_trial, g_test = TrialFunction(Vg), TestFunction(Vg)
    
    # Define variational problem
    # weak form for setting up the gradient equation
    a_g = inner(g_trial, g_test) * dx
    L_g = inner(rhs, g_test)*dx
    
    # Assemble linear system
    A_g, b_g = assemble_system(a_g, L_g, bcs_g)
    
    g_i = Function(Vg)
    # solve for gradient
    solve(A_g, g_i.vector(), b_g)
    
    return g_i


def evol_adjoint_solver_with_AS(adjoints, active_set_lb, active_set_ub, lam_D, states, data, savedir, the_domain, para):
    """Solve the adjoint equations with the given the active-set
     """
    [nt, dt, T] = get_time_partition_info(para)
    [gamma, rho] = get_const_info(para)

    V = adjoints[T].function_space()
    d, u, lam_pre = Function(V), Function(V), Function(V) # Current time-step variables

    i = 0
    #----------------------------------------------------
    while i <= nt:
        t = float(i*dt)

        # Update the current displacement
        u.assign(states[t])

        # Update data function
        d.assign(data[t])

        # Boundary Condition at the current time step
        lam_D.t = t

        # Associated adjoint RHS
        f = gamma*(u-d)
            
        # bounds for adjoint
        [lam_min, lam_max] = get_adj_bounds(V, active_set_lb[t], active_set_ub[t])
        
        # Solve associated adjoint equation
        lam = adjoint_solver(V, lam_D, lam_min, lam_max, f, the_domain, para)
        
        # update the state varialbes
        adjoints[t].assign(lam)
        
        # Update time
        i+=1
    
    return
    


def evol_adjoint_solver_forward(adjoints, lam_D, states, data, savedir, the_domain, para):
    """
    Solve the adjoint equations in order to determine the gradient of the objective function and, hence, the active-set
     """
    [nt, dt, T] = get_time_partition_info(para)
    [gamma, rho] = get_const_info(para)

    V = adjoints[T].function_space()
    d, u, lam_pre = Function(V), Function(V), Function(V) # Current time-step variables

    lam_min = interpolate(Constant(-1e6), V);
    lam_max = interpolate(Constant(1e6), V);
    
    i = 0
    #----------------------------------------------------
    while i <= nt:
        t = float(i*dt)

        # Update the current displacement
        u.assign(states[t])

        # Update data function
        d.assign(data[t])

        # Boundary Condition at the current time step
        lam_D.t = t

        # Associated adjoint RHS
        f = gamma*(u-d)
            
        # Solve associated adjoint equation
        lam = adjoint_solver(V, lam_D, lam_min, lam_max, f, the_domain, para)
        
        # update the state varialbes
        adjoints[t].assign(lam)
        
        # Update time
        i+=1
    
    return
    

def get_active_set(self):
    u_array = self.u.vector().get_local();
    ub_u_array = self.ub_u.vector().get_local();
    lb_u_array = self.lb_u.vector().get_local();
    # Find the active set and save it in a list, UAS
    self.UAS = list(u_array[:] >(ub_u_array[:] - 1e-8))
    self.LAS = list(u_array[:] <(lb_u_array[:] + 1e-8))
    return

def set_bounds_adj(self):
    ub_adj_array = self.ub_adj.vector().get_local();
    lb_adj_array = self.lb_adj.vector().get_local();
    #Set upper and lower bounds for adjoint
    for at_dof, is_active_point in enumerate(self.UAS): # for UAS
        if is_active_point == 1 :
            ub_adj_array[at_dof] = 0.; lb_adj_array[at_dof] = 0.;

    for at_dof, is_active_point in enumerate(self.LAS): # for LAS
        if is_active_point == 1 :
            ub_adj_array[at_dof] = 0.; lb_adj_array[at_dof] = 0.;
    self.ub_adj.vector()[:] = ub_adj_array;
    self.lb_adj.vector()[:] = lb_adj_array;
    self.problem_adj.set_bounds(self.lb_adj.vector(), self.ub_adj.vector())
    return


def get_evol_active_set(states, gradients, active_set_lb, active_set_ub, para):
    """Determine the active-set active_set_lb and active_set_ub"""
    V = states[0].function_space()
    
    [nt, dt, T] = get_time_partition_info(para)
    i = 0
    #----------------------------------------------------
    while i <= nt:
        if i == 0:
            t, t1 = float(i*dt), float((i + 1)*dt)
            lb = interpolate(Constant(0.), V);
            lb_u_array = lb.vector().get_local();
            ub_u_array = states[t1].vector().get_local();
            
        elif i == nt:
            t_1, t = float((i - 1)*dt), float(i*dt)
            ub = interpolate(Constant(1e6), V);
            ub_u_array = ub.vector().get_local();
            lb_u_array = states[t_1].vector().get_local();
        else:
            t_1, t, t1 = float((i - 1)*dt), float(i*dt), float((i + 1)*dt)
            ub_u_array = states[t1].vector().get_local();
            lb_u_array = states[t_1].vector().get_local();
            
        # get state arrays
        u_array = states[t].vector().get_local();
        gradients_array = gradients[t].vector().get_local();
        
        # get active set arrays
        LAS_array = active_set_lb[t].vector().get_local()
        UAS_array = active_set_ub[t].vector().get_local()
        
        # Find the active set and save it in a list, UAS
        cc = 1e10;
        LAS_array[:] = gradients_array[:] + cc*(lb_u_array[:] - u_array[:])
        UAS_array[:] = gradients_array[:] + cc*(ub_u_array[:] - u_array[:])
        
        LAS = list(LAS_array[:] > 0.)
        UAS = list(UAS_array[:] < 0.)
        
        active_set_lb[t].vector()[:] = LAS;
        active_set_ub[t].vector()[:] = UAS;
        
        # Update time
        i+=1
    #----------------------------------------------------
    return

    
def get_adj_bounds(V, active_set_lb_t, active_set_ub_t):

    LAS = active_set_lb_t.vector().get_local();
    UAS = active_set_ub_t.vector().get_local();
    
    adj_min = interpolate(Constant(-1e6), V);
    adj_max = interpolate(Constant(1e6), V);
    
    adj_max_array = adj_max.vector().get_local();
    adj_min_array = adj_min.vector().get_local();
    
    #Set upper and lower bounds for adjoint
    for at_dof, is_active_point in enumerate(LAS):
        if is_active_point == 1 :
            adj_max_array[at_dof] = 0.; adj_min_array[at_dof] = 0.;

    for at_dof, is_active_point in enumerate(UAS):
        if is_active_point == 1 :
            adj_max_array[at_dof] = 0.; adj_min_array[at_dof] = 0.;

    adj_max.vector()[:] = adj_max_array;
    adj_min.vector()[:] = adj_min_array;
    return [adj_min, adj_max]


def adjoint_solver(V, lam_D, lam_min, lam_max, f, the_domain, para):
    """
        """
    [gamma, rho] = get_const_info(para)
    
    [nt, dt, T] = get_time_partition_info(para)
    
    # boundary conditions
    bcs_adj = boundary_conditions_adj(V, the_domain, para);
    
    # define Trial and Test Functions
    lam_, lam_test, lam_trial = Function(V), TestFunction(V), TrialFunction(V)
    
    # weak form for setting up the state equation
    FF = -inner(grad(lam_), grad(lam_test))*dx + inner(f, lam_test)*dx
    
    HH = derivative(FF, lam_, TrialFunction(V))
    
    problem_adj = NonlinearVariationalProblem(FF, lam_, bcs=bcs_adj, J=HH)
    problem_adj.set_bounds(lam_min.vector(), lam_max.vector())
    
    solver_adj = NonlinearVariationalSolver(problem_adj)
    solver_adj.parameters.update(para["solver_u"])
    solver_adj.solve()
    return lam_
    

def ini_displacement_solver(states, u_D, ctrls, savedir, the_domain, para):
    """ Solver for the displacement at each time step with the lower bound is the displacement at the previous time step
        """
    [nt, dt, T] = get_time_partition_info(para)
    [gamma, rho] = get_const_info(para)

    V = states[T].function_space()
    
    #state constraints
    u_min = interpolate(Constant(0.), V);
    u_max = interpolate(Constant(1e6), V);
    
    #control variable
    f = Function(V)
    
    i = 1
    #----------------------------------------------------
    while i <= nt:
        t  = float(i*dt)

        # Update the current control
        f.assign(ctrls[t])
            
        # Boundary Condition at the current time step
        u_D.t = t
        
        # Solve associated displacement equation
        u = displacement_solver(V, u_D, u_min, u_max, f, the_domain, para)
        
        # update the state varialbes
        states[t].assign(u)
        
        # Update time
        i+=1
    
    return


def evol_displacement_solver_with_AS(states, u_D, ctrls, active_set_lb, active_set_ub, savedir, the_domain, para):
    """ Solver for the displacement at each time step with the lower bound is the displacement at the previous time step if the points are in the active-set associated to the lower bound, and upper bound is the displacement of the next time step if the points are in the active-set associated to the upper bound
        """
    [nt, dt, T] = get_time_partition_info(para)
    [gamma, rho] = get_const_info(para)

    V = states[T].function_space()
    
    active_set_lb_t, active_set_ub_t = Function(V), Function(V)
    u_pre, u_next = Function(V), Function(V)
    #control variable
    f = Function(V)
    
    i = nt
    #----------------------------------------------------
    while i >= 0:
        # get displacement of the previous and next time steps
        if i == 0:
            t, t_next  = float(i*dt), float((i+1)*dt)
            u_pre = interpolate(Constant(0.), V);
            u_next.assign(states[t_next])
        elif i == nt:
            t_pre, t = float((i-1)*dt), float(i*dt)
            u_pre.assign(states[t_pre])
            u_next = interpolate(Constant(1e6), V);
        else:
            t_pre, t, t_next  = float((i-1)*dt), float(i*dt), float((i+1)*dt)
            u_pre.assign(states[t_pre])
            u_next.assign(states[t_next])
            
        # get the current active-set
        active_set_lb_t.assign(active_set_lb[t])
        active_set_ub_t.assign(active_set_ub[t])
        
        # Update the current control
        f.assign(ctrls[t])
            
        # Boundary Condition at the current time step
        u_D.t = t
        
        # get bounds for the current time step
        [ui_min, ui_max] = get_current_pde_bounds(V, u_pre, u_next, active_set_lb_t, active_set_ub_t)
        
        # Solve associated displacement equation
        u = displacement_solver(V, u_D, ui_min, ui_max, f, the_domain, para)
        
        # update the state varialbes
        states[t].assign(u)
        
        # Update time
        i-=1
    
    return



def evol_displacement_solver(states, u_D, ctrls, savedir, the_domain, para):
    """ Solver for the displacement at each time step with the given control
        """
    [nt, dt, T] = get_time_partition_info(para)
    [gamma, rho] = get_const_info(para)

    V = states[T].function_space()
    
    #state constraints
    u_min = interpolate(Constant(0.), V);
    u_max = interpolate(Constant(1e6), V);
    
    #control variable
    f = Function(V)
    
    i = 0
    #----------------------------------------------------
    while i <= nt:
        t  = float(i*dt)
                
        # Update the current control
        f.assign(ctrls[t])
            
        # Boundary Condition at the current time step
        u_D.t = t
        
        # Solve associated displacement equation
        u = displacement_solver(V, u_D, u_min, u_max, f, the_domain, para)
        
        # update the state varialbes
        states[t].assign(u)
        
        # Update time
        i+=1
    return



def displacement_solver(V, u_D, u_min, u_max, f, the_domain, para):
    """
        """
    # boundary conditions
    bcs_u = boundary_conditions_4u(V, u_D, the_domain, para);
    
    # define Trial and Test Functions
    u, u_test = Function(V), TestFunction(V)
    
    # weak form for setting up the state equation
    F = inner(grad(u), grad(u_test))*dx - inner(f, u_test)*dx
    H = derivative(F, u, TrialFunction(V))
    
    problem_pde = NonlinearVariationalProblem(F, u, bcs=bcs_u, J=H)
    problem_pde.set_bounds(u_min.vector(), u_max.vector())
    
    solver_pde = NonlinearVariationalSolver(problem_pde)
    solver_pde.parameters.update(para["solver_u"])
    solver_pde.solve()
    return u

def boundary_conditions_4u(Vu, u0, the_domain, para):
    [mesh, boundary_markers, sub_domains, n, ds] = the_domain
    [sub_left, sub_top, sub_right, sub_bottom] = sub_domains
    bc_type = para["problem"]["setting"]["bc_type"]
    
    if bc_type == 1: #two_side clamped
        # PDE-Dirichlet B.C. for displacement field
        bcl_u = DirichletBC(Vu, u0, sub_left);
        bcr_u = DirichletBC(Vu, u0, sub_right);
        bcs_u = [bcl_u, bcr_u]
    elif bc_type == 2: #all clamped
        bcl_u = DirichletBC(Vu, u0, sub_left);
        bct_u = DirichletBC(Vu, u0, sub_top);
        bcr_u = DirichletBC(Vu, u0, sub_right);
        bcb_u = DirichletBC(Vu, u0, sub_bottom);
        bcs_u = [bcl_u, bct_u, bcr_u, bcb_u]
    return bcs_u


def boundary_conditions_adj(Vu, the_domain, para):
    [mesh, boundary_markers, sub_domains, n, ds] = the_domain
    [sub_left, sub_top, sub_right, sub_bottom] = sub_domains
    bc_type = para["problem"]["setting"]["bc_type"]

    if bc_type == 1:
        # PDE-Dirichlet B.C. for displacement field
        bcl_u = DirichletBC(Vu, Constant(0.), sub_left);
        bcr_u = DirichletBC(Vu, Constant(0.), sub_right);
        bcs_u = [bcl_u, bcr_u]
    elif bc_type == 2: #all clamped
        bcl_u = DirichletBC(Vu, Constant(0.), sub_left);
        bct_u = DirichletBC(Vu, Constant(0.), sub_top);
        bcr_u = DirichletBC(Vu, Constant(0.), sub_right);
        bcb_u = DirichletBC(Vu, Constant(0.), sub_bottom);
        bcs_u = [bcl_u, bct_u, bcr_u, bcb_u]
    return bcs_u
    
def create_mesh(para, savedir):
    [case, bc_type, Lx, Ly, cell_size] = get_setting_info(para);
    Nx = int(Lx/cell_size); Ny = int(Ly/cell_size);
    
    mesh = RectangleMesh(Point(0,0), Point(Lx, Ly), Nx, Ny)
    
    # Define the boundaries
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0., DOLFIN_EPS)
    
    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Ly, DOLFIN_EPS)
    
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], Lx, DOLFIN_EPS)
    
    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0., DOLFIN_EPS)
    
    sub_left = Left(); sub_top = Top();
    sub_right = Right(); sub_bottom = Bottom();
    
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0) # marked as 0
    sub_left.mark(boundary_markers, 1) # marked as 1
    sub_top.mark(boundary_markers, 2) # marked as 2
    sub_right.mark(boundary_markers, 3) # marked as 3
    sub_bottom.mark(boundary_markers, 4) # marked as 4
    sub_domains = [sub_left, sub_top, sub_right, sub_bottom]
    
    #hmax = mesh.hmax(); hmin = mesh.hmin();
    #num_cells = mesh.num_cells(); num_vertices = mesh.num_vertices();
    
    # Save boundary markers
    File(savedir + "/mesh/boundary_markers%s.pvd"%case) << boundary_markers
    # Redefine boundary integration measure .NOTE: ds is for external bounray. dS for internal boundary
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    n = FacetNormal(mesh) # the facet normal vector
    the_domain = [mesh, boundary_markers, sub_domains, n, ds]
    return the_domain
    


def default_parameters():
    
    para = Parameters("para");
    para.add("ex", 1)
    para.parse();
    example = para["ex"]
    
    if example == 2:
        problem_info = eval("default_example_two_parameters()");
    elif example == 3:
        problem_info = eval("default_example_three_parameters()");
    elif example == 4:
        problem_info = eval("default_example_four_parameters()");
    else:
        problem_info = eval("default_example_one_parameters()");
    para.add(problem_info)
    
    # Evaluate other parameters
    subset_list = ["initial_guess", "const", "time_partition", "GDA", "solver_u"];
    for subparset in subset_list:
        subparset_is = eval("default_" + subparset + "_parameters()");
        para.add(subparset_is);
    
    return para
    

def default_example_one_parameters():
    setting = Parameters("setting");
    setting.add("bc_type", 1);# bc type: 1- two-side clamped; 2-all clamped
    setting.add("Lx", 1.);
    setting.add("Ly", 0.5);
    setting.add("cell_size", 0.01);
    problem = Parameters("problem");
    problem.add("case", 100)
    problem.add(setting);
    return problem


def default_example_four_parameters():
    setting = Parameters("setting");
    setting.add("bc_type", 2);# bc type: 1- two-side clamped; 2-all clamped
    setting.add("Lx", 1.);
    setting.add("Ly", 1.);
    setting.add("cell_size", 0.01);
    problem = Parameters("problem");
    problem.add("case", 400)
    problem.add(setting);
    return problem


def default_example_three_parameters():
    setting = Parameters("setting");
    setting.add("bc_type", 1);# bc type: 1- two-side clamped; 2-all clamped
    setting.add("Lx", 1.);
    setting.add("Ly", 0.5);
    setting.add("cell_size", 0.01);
    problem = Parameters("problem");
    problem.add("case", 300)
    problem.add(setting);
    return problem


def default_example_two_parameters():
    setting = Parameters("setting");
    setting.add("bc_type", 2);# bc type: 1- two-side clamped; 2-all clamped
    setting.add("Lx", 1.);
    setting.add("Ly", 0.5);
    setting.add("cell_size", 0.01);
    problem = Parameters("problem");
    problem.add("case", 200)
    problem.add(setting);
    return problem

def get_setting_info(para):
    case = para["problem"]["case"];
    bc_type = para["problem"]["setting"]["bc_type"];
    Lx = para["problem"]["setting"]["Lx"];
    Ly = para["problem"]["setting"]["Ly"];
    cell_size = para["problem"]["setting"]["cell_size"];
    
    setting_info = [case, bc_type, Lx, Ly, cell_size];
    return setting_info

def default_initial_guess_parameters():
    initial_guess = Parameters("initial_guess");
    initial_guess.add("is_constant", True);
    initial_guess.add("slope", 100.);
    initial_guess.add("noise_level", 0.0);
    initial_guess.add("from_manufactured_sol", False);
    
    return initial_guess
    
def get_initial_guess_info(para):
    is_constant = para["initial_guess"]["is_constant"];
    slope = para["initial_guess"]["slope"];
    noise_level = para["initial_guess"]["noise_level"];
    from_manufactured_sol = para["initial_guess"]["from_manufactured_sol"];

    initial_guess_info = [is_constant, slope, noise_level, from_manufactured_sol];
    return initial_guess_info

def default_const_parameters():
    const = Parameters("const");
    const.add("gamma", 1.);
    const.add("rho", 1.e-15);
    return const
    

def get_const_info(para):
    gamma = para["const"]["gamma"];
    rho = para["const"]["rho"];
    const_info = [gamma, rho];
    return const_info
    
def default_time_partition_parameters():
    time_partition = Parameters("time_app");
    T = 1.; # final time
    nt = 20; # the number of time steps
    time_partition.add("T", T);
    time_partition.add("nt", nt);
    return time_partition
    
def get_time_partition_info(para):
    T = para["time_app"]["T"];
    nt = para["time_app"]["nt"];
    dt = float(T/nt)
    time_partition_info = [nt, dt, T];
    return time_partition_info
    

def default_GDA_parameters():
    GDA = Parameters("GDA");
    GDA = Parameters("GDA");
    GDA.add("iter_max", 1000);
    GDA.add("save_frequency", 100);
    GDA.add("tol", 1.e-4);
    GDA.add("step_size", 1.);
    GDA.add("step_size_min", 1.e-3);
    GDA.add("step_size_max", 10.);
    GDA.add("search_scheme", "non_constant_step_size"); #
    return GDA

def get_GDA_info(para):
    iter_max = para["GDA"]["iter_max"];
    save_frequency = para["GDA"]["save_frequency"];
    tol = para["GDA"]["tol"];
    step_size = para["GDA"]["step_size"];
    step_size_min = para["GDA"]["step_size_min"];
    step_size_max = para["GDA"]["step_size_max"];
    search_scheme = para["GDA"]["search_scheme"];
    GDA_info = [iter_max, save_frequency, tol, step_size, step_size_min, step_size_max, search_scheme];
    return GDA_info

def default_solver_u_parameters():
    
    krylov_solver = Parameters("krylov_solver")
    krylov_solver.add("absolute_tolerance", 1.e-12);
    krylov_solver.add("divergence_limit", 1.e4);
    krylov_solver.add("error_on_nonconvergence", True );
    krylov_solver.add("maximum_iterations", 1000);
    krylov_solver.add("monitor_convergence", False);
    krylov_solver.add("nonzero_initial_guess", False);
    krylov_solver.add("relative_tolerance", 1.e-12);
    krylov_solver.add("report", True);
    
    lu_solver = Parameters("lu_solver")
    lu_solver.add("report", True );
    lu_solver.add("symmetric", False);
    lu_solver.add("verbose", False);
    
    snes_solver = Parameters("snes_solver");
    snes_solver.add("absolute_tolerance", 1.e-10);
    snes_solver.add("error_on_nonconvergence", True);
    snes_solver.add("line_search", "basic");
    snes_solver.add("linear_solver", "lu");
    snes_solver.add("maximum_iterations", 50);
    snes_solver.add("maximum_residual_evaluations", 200);
    snes_solver.add("method", "vinewtonrsls");
    snes_solver.add("preconditioner", "default");
    snes_solver.add("relative_tolerance", 1.e-10);
    snes_solver.add("report", False);
    snes_solver.add("sign", "default");
    snes_solver.add("solution_tolerance", 1.e-10);
    snes_solver.add(krylov_solver);
    snes_solver.add(lu_solver);
    
    solver_u = Parameters("solver_u");
    solver_u.add("nonlinear_solver", "snes"); #[newton,snes]
    solver_u.add("print_matrix", False);
    solver_u.add("print_rhs", False);
    solver_u.add("symmetric", False);
    solver_u.add(snes_solver);
    return solver_u
    
if __name__== "__main__":
    start = time. time()
    main()
    # we can test the solver by running this function or using pytest function i.e. >> python3 -m pytest time_adjoint_method.py
#    test_first_div_app()
#    test_second_div_app()

    end = time. time()
    print("\nExecution time (in seconds):",(end - start))
