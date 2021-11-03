"""
    Phase-field Approach to Optimal Design with Obstacle
    Constraints
    
    default parameters for the problem without obstacle
    """

from fenics import *
#from dolfin import *
#from dolfin_adjoint import *
def default_parameters():
    
    para = Parameters("para");
    # ex: 1. no_obstacle , 2. one_obstacle (default), or 3. two_obstacles
    para.add("ex", 2)
    para.parse();
    example = para["ex"]
    if example == 1:
        problem_info = eval("default_no_obstacle_parameters()");
    elif example == 3:
        problem_info = eval("default_two_obstacle_parameters()");
    else :
        problem_info = eval("default_one_obstacle_parameters()");
    para.add(problem_info)
    
    # Evaluate other parameters
    subset_list = ["mesh", "material", "post_processing", "solver_alpha", "solver_u"];
    for subparset in subset_list:
        subparset_is = eval("default_" + subparset + "_parameters()");
        para.add(subparset_is);
    
#    print("\n Print the default parameters to screen \n", para.str(True))
    return para

def default_no_obstacle_parameters():
    
    initial_guess = Parameters("initial_guess");
    initial_guess.add("is_constant", True);
    initial_guess.add("alpha_0", 1.);
    initial_guess.add("sin_px", 1); # period in x-direction
    initial_guess.add("sin_py", 1); # period in y-direction
    
    force = Parameters("force");
    force.add("f", 0.); # body force
    force.add("gx", 0.); # traction force in x-direction
    force.add("gy", -1.); # traction force in y-direction
    
    obstacle = Parameters("obstacle");
    obstacle.add("ub_u_x", 100.);
    obstacle.add("ub_u_y", 100.);
    obstacle.add("lb_u_x", -100.);
    obstacle.add("lb_u_y", -100.);
    obstacle.add("ub_alpha", 1.);
    obstacle.add("lb_alpha", 0.);
    
    const = Parameters("const");
    const.add("p", 2.);
    const.add("q", 1.);
    const.add("w_type", 1);
    const.add("delta", 1.e-3);
    const.add("eta", 0.08);
    const.add("kappa", 5.e-5);
    
    problem = Parameters("problem");
    problem.add("case", 0)
    problem.add(initial_guess);
    problem.add(force);
    problem.add(obstacle);
    problem.add(const);
    return problem

def default_one_obstacle_parameters():
    
    initial_guess = Parameters("initial_guess");
    initial_guess.add("is_constant", True);
    initial_guess.add("alpha_0", 1.);
    initial_guess.add("sin_px", 1); # period in x-direction
    initial_guess.add("sin_py", 1); # period in y-direction
    
    force = Parameters("force");
    force.add("f", 0.); # body force
    force.add("gx", 0.); # traction force in x-direction
    force.add("gy", -1.); # traction force in y-direction
    
    obstacle = Parameters("obstacle");
    obstacle.add("ub_u_x", 1.e8);
    obstacle.add("ub_u_y", 1.e8);
    obstacle.add("lb_u_x", -1.e8);
    obstacle.add("lb_u_y", -0.02);
    obstacle.add("ub_alpha", 1.);
    obstacle.add("lb_alpha", 0.);
    
    const = Parameters("const");
    const.add("p", 2.);
    const.add("q", 1.);
    const.add("w_type", 1);
    const.add("delta", 1.e-3);
    const.add("eta", 0.08);
    const.add("kappa", 5.e-5);
    
    problem = Parameters("problem");
    problem.add("case", 0)
    problem.add(initial_guess);
    problem.add(force);
    problem.add(obstacle);
    problem.add(const);
    return problem

def default_two_obstacle_parameters():
    
    initial_guess = Parameters("initial_guess");
    initial_guess.add("is_constant", True);
    initial_guess.add("alpha_0", 1.);
    initial_guess.add("sin_px", 1); # period in x-direction
    initial_guess.add("sin_py", 1); # period in y-direction
    
    force = Parameters("force");
    force.add("f", 0.); # body force
    force.add("gx", 0.); # traction force in x-direction
    force.add("gy", -10.); # traction force in y-direction
    
    obstacle = Parameters("obstacle");
    obstacle.add("ub_u_x", 1.e8);
    obstacle.add("ub_u_y", 0.25);
    obstacle.add("lb_u_x", -1.e8);
    obstacle.add("lb_u_y", -0.035);
    obstacle.add("ub_alpha", 1.);
    obstacle.add("lb_alpha", 0.);
    
    const = Parameters("const");
    const.add("p", 2.);
    const.add("q", 1.);
    const.add("w_type", 1);
    const.add("delta", 1.e-3);
    const.add("eta", 0.08);
    const.add("kappa", 5.e-5);
    
    problem = Parameters("problem");
    problem.add("case", 0)
    problem.add(initial_guess);
    problem.add(force);
    problem.add(obstacle);
    problem.add(const);
    return problem


def default_mesh_parameters():
    
    mesh = Parameters("mesh");
    mesh.add("cell_size", 0.005);
    mesh.add("structured_mesh", True);
    mesh.add("resolution", 85);
    mesh.add("Lx", 1.);
    mesh.add("Ly", 0.3);
    return mesh

def default_material_parameters():
    
    material = Parameters("material");
    material.add("E", 1.e3);
    material.add("nu", 0.3);
    material.add("Gc", 1.);
    material.add("ell", 0.025); #5 times cell_size
    material.add("cw", 1.);
    return material

def default_solver_alpha_parameters():
    
    krylov_solver = Parameters("krylov_solver")
    krylov_solver.add("absolute_tolerance", 1.e-12);
    krylov_solver.add("divergence_limit", 1.e4);
    krylov_solver.add("error_on_nonconvergence", False );
    krylov_solver.add("maximum_iterations", 1000);
    krylov_solver.add("monitor_convergence", True);
    krylov_solver.add("nonzero_initial_guess", False);
    krylov_solver.add("relative_tolerance", 1.e-12);
    krylov_solver.add("report", True);
    
    # method = tao_type: tron blmvm gpcg pounders;
    # linesearch: default (more-thuente), unit, more-thuente, gpcg, armijo, owarmijo, ipm;
    # linear_solver = ksp_type: nash, lu
    # preconditioner = pc_type: ml_amg
    solver_alpha = Parameters("solver_alpha");
    solver_alpha.add("error_on_nonconvergence", True);
    solver_alpha.add("gradient_absolute_tol", 5.e-8);
    solver_alpha.add("gradient_relative_tol", 5.e-8);
    solver_alpha.add("gradient_t_tol", 5.e-8);
    solver_alpha.add("line_search", "default");
    solver_alpha.add("linear_solver", "lu");
    solver_alpha.add("maximum_iterations", 2000);
    solver_alpha.add("method", "blmvm");
    solver_alpha.add("monitor_convergence", True);
    solver_alpha.add("options_prefix", "myTAO");
    solver_alpha.add("preconditioner", "bjacobi");
    solver_alpha.add("report", True);
    solver_alpha.add(krylov_solver);
    return solver_alpha

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
    
#    newton_solver = Parameters("newton_solver");
#    newton_solver.add("absolute_tolerance", 1.e-10);
#    newton_solver.add("convergence_criterion", "residual");
#    newton_solver.add("error_on_nonconvergence", False);
#    newton_solver.add("linear_solver", "default");
#    newton_solver.add("maximum_iterations", 50);
#    newton_solver.add("preconditioner", "default");
#    newton_solver.add("relative_tolerance", 1.e-10);
#    newton_solver.add("relaxation_parameter", "default");
#    newton_solver.add("report", False);
#    newton_solver.add(krylov_solver);
#    newton_solver.add(lu_solver);
#    
    solver_u = Parameters("solver_u");
    solver_u.add("nonlinear_solver", "snes"); #[newton,snes]
    solver_u.add("print_matrix", False);
    solver_u.add("print_rhs", False);
    solver_u.add("symmetric", False);
    solver_u.add(snes_solver);
#    solver_u.add(newton_solver);

    return solver_u

def default_post_processing_parameters():
    
    post_processing = Parameters("post_processing");
    # Print TAO evoluiton in one processor to examine the results in a clearer way.
    # useful for debugging
    post_processing.add("TAO_print_function_val", False);
    post_processing.add("TAO_print_gradient_val", False);
    
    post_processing.add("TAO_saving_frequency", 200);
    post_processing.add("TAO_save_pvd_u", True);
    post_processing.add("TAO_save_pvd_u_adj", True);
    post_processing.add("TAO_save_pvd_alpha", True);
    
    post_processing.add("TAO_save_xdmf_u", False);
    post_processing.add("TAO_save_xdmf_u_adj", False);
    post_processing.add("TAO_save_xdmf_alpha", False);

    post_processing.add("save_opt_u", True);
    post_processing.add("save_opt_active_set_ub", True);
    post_processing.add("save_opt_active_set_lb", True);
    post_processing.add("save_opt_u_adj", True);
    post_processing.add("save_opt_alpha", True);
    post_processing.add("save_opt_stress", True);
    return post_processing

def get_post_processing_parameters(para, option):
    
    TAO_print_function_val = para["post_processing"]["TAO_print_function_val"];
    TAO_print_gradient_val = para["post_processing"]["TAO_print_gradient_val"];
    
    TAO_saving_frequency = para["post_processing"]["TAO_saving_frequency"];
    TAO_save_pvd_u = para["post_processing"]["TAO_save_pvd_u"];
    TAO_save_pvd_u_adj = para["post_processing"]["TAO_save_pvd_u_adj"];
    TAO_save_pvd_alpha = para["post_processing"]["TAO_save_pvd_alpha"];
    
    TAO_save_xdmf_u = para["post_processing"]["TAO_save_xdmf_u"];
    TAO_save_xdmf_u_adj = para["post_processing"]["TAO_save_xdmf_u_adj"];
    TAO_save_xdmf_alpha = para["post_processing"]["TAO_save_xdmf_alpha"];
    
    save_opt_u = para["post_processing"]["save_opt_u"];
    save_opt_active_set_ub = para["post_processing"]["save_opt_active_set_ub"];
    save_opt_active_set_lb = para["post_processing"]["save_opt_active_set_lb"];
    save_opt_u_adj = para["post_processing"]["save_opt_u_adj"];
    save_opt_alpha = para["post_processing"]["save_opt_alpha"];
    save_opt_stress = para["post_processing"]["save_opt_stress"];
    
    if option == "TAO_pp":
        TAO_post_processing_parameters = [TAO_print_function_val, TAO_print_gradient_val, TAO_saving_frequency, TAO_save_pvd_u, TAO_save_pvd_u_adj, TAO_save_pvd_alpha, TAO_save_xdmf_u, TAO_save_xdmf_u_adj, TAO_save_xdmf_alpha]
        return TAO_post_processing_parameters
    elif option == "opt_pp":
        opt_post_processing_parameters = [save_opt_u, save_opt_active_set_ub, save_opt_active_set_lb, save_opt_u_adj, save_opt_alpha, save_opt_stress]
        return opt_post_processing_parameters
    else:
        return TAO_post_processing_parameters, opt_post_processing_parameters

def get_mesh_info(para):
    
    cell_size = para["mesh"]["cell_size"];
    structured_mesh = para["mesh"]["structured_mesh"];
    resolution = para["mesh"]["resolution"];
    Lx = para["mesh"]["Lx"];
    Ly = para["mesh"]["Ly"];
    mesh_info = [cell_size, Lx, Ly, structured_mesh, resolution];
    return mesh_info

def get_material_info(para):
    
    E = para["material"]["E"];
    nu = para["material"]["nu"];
    Gc = para["material"]["Gc"];
    ell = para["material"]["ell"];
    cw = para["material"]["cw"];
    material_info = [E, nu, Gc, ell, cw];
    return material_info


def get_initial_guess_info(para):

    is_constant = para["problem"]["initial_guess"]["is_constant"];
    alpha_0 = para["problem"]["initial_guess"]["alpha_0"];
    sin_px = para["problem"]["initial_guess"]["sin_px"];
    sin_py = para["problem"]["initial_guess"]["sin_py"];
    initial_guess_info = [is_constant, alpha_0, sin_px, sin_py];
    return initial_guess_info


def get_force_info(para):
    
    f = para["problem"]["force"]["f"];
    g_x = para["problem"]["force"]["gx"];
    g_y = para["problem"]["force"]["gy"];
    force_info = [f, g_x, g_y ]
    return force_info

def get_problem_constant_info(para):
    
    case = para["problem"]["case"];
    p = para["problem"]["const"]["p"];
    q = para["problem"]["const"]["q"];
    w_type = para["problem"]["const"]["w_type"];
    delta = para["problem"]["const"]["delta"];
    eta = para["problem"]["const"]["eta"];
    kappa = para["problem"]["const"]["kappa"];
    
    problem_constant_info = [case, p, q, w_type, delta, eta, kappa]
    return problem_constant_info

def get_problem_obstacle_info(para):
    ub_u_x = para["problem"]["obstacle"]["ub_u_x"];
    ub_u_y = para["problem"]["obstacle"]["ub_u_y"];
    lb_u_x = para["problem"]["obstacle"]["lb_u_x"];
    lb_u_y = para["problem"]["obstacle"]["lb_u_y"];
    ub_alpha = para["problem"]["obstacle"]["ub_alpha"];
    lb_alpha = para["problem"]["obstacle"]["lb_alpha"];
    problem_obstacle_info = [ub_u_x, ub_u_y, lb_u_x, lb_u_y, ub_alpha, lb_alpha];
    return problem_obstacle_info
    
