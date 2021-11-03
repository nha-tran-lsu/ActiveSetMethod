"""
    Phase-field Approach to Optimal Design with Obstacle
    Constraints
    \min_{J(\alpha, u(\alpha)) =  \int_{\Omega} f u dx + \int_{\partial\Omega} g u ds + \eta \int_{\Omega} \alpha^q dx} + \kappa \frac{Gc}{2*c_w} \int_\Omega \frac{w(alpha)}{\ell} + \ell \nabla(\alpha)\cdot\nabla(\alpha) dx
    s.t
    subject to the variational inequality constraint
    
    -div(\Phi(\alpha) * A * e(u)) = f     , in \Omega
    u = uD    , on \partial \Omega.
    \Phi(\alpha) * A * e(u) . n = g     , on \partial_D \Omega.
    u_min <= u <= u_max
    \phi(\alpha) = (1-\delta)*\alpha^p + \delta
    
    delta > 0, a small number
    FEniCS program: Active-set Algorithm
    """


from __future__ import print_function

from fenics import *
from dolfin import *
from mshr import *
from user_parameters import *
import sys, os, sympy, math, time
import petsc4py as PETSc
import numpy as np
#import matplotlib.pyplot as plt


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


def main():
    # Optimization options for the form compiler
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"

    mpi_comm = MPI.comm_world; my_rank = MPI.rank(mpi_comm)
    mpi_size = MPI.size(mpi_comm);
    pinfty = np.inf; ninfty = -np.inf;
    
    # Pick an example:
    # 1. no_obstacle , 2. one_obstacle (default), or 3. two_obstacles
    para = default_parameters(); # load the default parameters
    para.parse(); # take inputs from the command line
    example = para["ex"]
    
    # Get the parameteres information for the_example
    [cell_size, Lx, Ly, structured_mesh, resolution] = get_mesh_info(para);
    [E, nu, Gc, ell, cw] = get_material_info(para);
    [case, p, q, w_type, delta, eta, kappa] = get_problem_constant_info(para);
    Nx = int(Lx/cell_size); Ny = int(Ly/cell_size);
    
    # Elasticity parameters
    mu, lambda_ = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
    model = "plane_stress"
    if model == "plane_stress":
        lambda_ = 2*mu*lambda_/(lambda_+2*mu)
    elas_para = [mu, lambda_]

    # Set directory (read mesh and then define later)
    if example == 1:
        prefix = "no_obstacle" + "/AS_%s%s_"%(example, case)
    elif example == 3:
        prefix = "two_obstacles" + "/AS_%s%s_"%(example, case)
    else:
        prefix = "one_obstacle" + "/AS_%s%s_"%(example, case)

    savedir = "AS_phase_field_results_2021_delta_%s/%s_cell_size%s_Nx%s_Ny%s_Lx%s_Ly%sp%s_q%s_"%(delta, prefix, cell_size, Nx, Ny, Lx, Ly, p, q)
    
    # Create mesh
    [mesh, boundary_markers, sub_domains, ds] = create_mesh(para, savedir);
    the_domain = [mesh, boundary_markers, sub_domains, ds]
    meshname = savedir + "/mesh%s/mesh%s.xdmf"%(case, case)
    XDMFFile(MPI.comm_world, meshname).write(mesh)
    hmax = mesh.hmax(); hmin = mesh.hmin();
    num_cells = mesh.num_cells(); num_vertices = mesh.num_vertices();

    # save the parameters used for the code
    File(savedir + "/parameters%s_%s.xml"%(example, case)) << para
    # Print parameters
    print("\n The parameters: \n", para.str(False));
    print("\n hmax = %f; hmin = %f; num_cells = %f; num_vertices = %f" %(hmax, hmin, num_cells, num_vertices));

    # Define function spaces (FS)
    V_pde = VectorFunctionSpace(mesh, 'P', 2) # Displacement FS
    V_adj = VectorFunctionSpace(mesh, 'P', 2) # Lagrange multiplier FS
    V_alpha = FunctionSpace(mesh, 'P', 1) # Alpha FS

    # Boundary conditions
    bcs_pde, bcs_adj, bcs_alpha = boundary_condition(V_pde, V_adj, V_alpha, sub_domains, example);
    bcs = [bcs_pde, bcs_adj, bcs_alpha];

    # Get forces
    [f, g_L1, g_L2, g_T3, g_T4, g_R5, g_R6, g_R7, g_B8, g_B9] = get_forces(example, para);
    forces = [f, g_L1, g_L2, g_T3, g_T4, g_R5, g_R6, g_R7, g_B8, g_B9];

    # Define solution variables
    u, v, du  = Function(V_pde), TestFunction(V_pde), TrialFunction(V_pde)
    u_adj = Function(V_adj);
    alpha = Function(V_alpha);
    d = u.geometric_dimension()  # space dimension

    # define the active sets
    active_set_lb = Function(V_pde) # associated to the lower bound
    active_set_ub = Function(V_pde) # associated to the upper bound
    
    # Initial guess for alpha
    alpha_0 = alpha_initial_guess(V_alpha, para, Lx, Ly);
    alpha.assign(interpolate(alpha_0, V_alpha))
    File(savedir + "/opt_sol%s/alpha_initial%s.pvd"%(case, case)) << alpha
    
    # Get obstacle constraints
    ub_u, lb_u, ub_alpha, lb_alpha = get_obstacle_constraints(example, para, V_pde, V_alpha, savedir)
    obstacles = [ub_u, lb_u, ub_alpha, lb_alpha]

    # Get the active-set object
    problem_alpha = active_set_object(u, u_adj, alpha, active_set_lb, active_set_ub, the_domain, bcs, forces, obstacles, para, elas_para, savedir)

    # Set solver parameters for TAO
    solver_TAO_alpha = PETScTAOSolver()
    solver_TAO_alpha.parameters.update(para["solver_alpha"])
    # solver_TAO_alpha.parameters.str(True)

    iter = solver_TAO_alpha.solve(problem_alpha, alpha.vector(), lb_alpha.vector(), ub_alpha.vector())

    # Get the optimal solutions
    u_opt = problem_alpha.u
    u_adj_opt = problem_alpha.u_adj
    alpha_opt = problem_alpha.alpha
    active_set_lb_opt = problem_alpha.active_set_lb
    active_set_ub_opt = problem_alpha.active_set_ub

    # Get the number of times the PDE solver is executed
    count_PDE_execution = problem_alpha.count

    # Save the optimal solutions
    save_opt_sol(alpha_opt, u_opt, u_adj_opt, active_set_lb_opt, active_set_ub_opt, para, savedir, mesh)

    #--------------------------------------
    # Print some results to the screen
    work = assemble(dot(f,u_opt)*dx + dot(g_L1, u_opt)*ds(1)+ dot(g_L2, u_opt)*ds(2) + dot(g_T3, u_opt)*ds(3) + dot(g_T4, u_opt)*ds(4) + dot(g_R5,u_opt)*ds(5) + dot(g_R6,u_opt)*ds(6) + dot(g_R7,u_opt)*ds(7) + dot(g_B8,u_opt)*ds(8) + dot(g_B9,u_opt)*ds(9));
    compliance = assemble(eta*alpha_opt**q*dx);
    perimeter_penalization = assemble(kappa*Gc/(2. * cw) * (w(alpha_opt, w_type)/ell + ell * inner(grad(alpha_opt), grad(alpha_opt)))*dx);
    vol_alpha_opt = assemble(alpha_opt*dx)
    vol_ratio_percentage = assemble(alpha_opt*dx)/(1*Lx*Ly)*100.;
    u1, u2 = u_opt.split(True)
    alpha_opt_max = alpha_opt.vector().max()
    alpha_opt_min = alpha_opt.vector().min()
    alpha_opt_norm = norm(alpha_opt)

    # summary results to a text file
    write_results = open("summary_AS_results_0921_ex%s%s_mesh_idp.txt"%(example, case), "a+")
    write_results.write("\n Summary results of example {:d}, case {:d}" .format(example, case))
    write_results.write("\n Objective function = {:.10f}" .format (work + compliance + perimeter_penalization) )
    write_results.write("\n The number of times the PDE solver is executed: count = {:d}" .format (count_PDE_execution) )
    write_results.write("\n Work_plus_compliance = {:.10f}" .format(work+compliance) )
    write_results.write("\n Work = {:.10f} \r\n" .format(work) )
    write_results.write("\n Compliance = {:.10f}" .format(compliance) )
    write_results.write("\n Perimeter_penalization = {:.10f}" .format(perimeter_penalization) )
    write_results.write("\n Vol_alpha = {:.10f}" .format(vol_alpha_opt) )
    write_results.write("\n Vol_ratio_percentage = {:.10f}" .format(vol_ratio_percentage) )
    write_results.write("\n Alpha_norm = {:.10f}"  .format(alpha_opt_norm) )
    write_results.close()

 #   print to screen some results
    print("\nwork = {:.8f}; compliance = {:.10f}" .format(work, compliance))
    print("\nperimeter_penalization = {:.10f}" .format(perimeter_penalization))
    print("Norm of u_opt: %s" % sqrt(assemble(inner(u_opt, u_opt)*dx)))
    print("Norm of alpha_opt: %s" % alpha_opt_norm)
    print("\n vol_alpha = {:.10f}" .format(vol_alpha_opt))
    print("\n vol_ratio_percentage = {:.5f}" .format(vol_ratio_percentage))
    print("Max of alpha_opt: %s " % alpha_opt_max)
    print("Min of alpha_opt: %s " % alpha_opt_min)
    print("min/max u_opt_x:", u1.vector().get_local().min(), u1.vector().max())
    print("min/max u_opt_y:", u2.vector().get_local().min(), u2.vector().max())

    # Save the optimal stress
    save_opt_stress = para["post_processing"]["save_opt_stress"];
    if save_opt_stress:
        def sigma(u):#sigma(u, alpha, p, delta, mu, lambda_, d):
            return phi(alpha_opt, p, delta)* sigma_0(u, mu, lambda_, d)

        V_sig = TensorFunctionSpace(mesh, "DG", degree=0)
        stress = project(sigma(u_opt), V=V_sig)
        File(savedir + "/opt_sol%s/stress%s.pvd"%(case, case)) << stress
    return

class active_set_object(OptimisationProblem):
    def __init__(self, u, u_adj, alpha, active_set_lb, active_set_ub, the_domain, bcs, forces, obstacles,para, elas_para, savedir):
        # Call the parent class OptimisationProblem
        OptimisationProblem.__init__(self)
        # Some object's attributes
        self.para = para;
        self.u = u; V_pde = self.u.function_space();
        self.u_adj = u_adj; V_adj = self.u_adj.function_space();
        self.alpha = alpha; V_alpha = self.alpha.function_space();
        self.active_set_lb = active_set_lb;
        self.active_set_ub = active_set_ub;
        
        [mesh, boundary_markers, sub_domains, ds] = the_domain

        [bcs_pde, bcs_adj, bcs_alpha] = bcs
        self.bcs_pde = bcs_pde; self.bcs_adj = bcs_adj; self.bcs_alpha = bcs_alpha;

        [f_pde, g_L1, g_L2, g_T3, g_T4, g_R5, g_R6, g_R7, g_B8, g_B9] = forces

        [ub_u, lb_u, ub_alpha, lb_alpha] = obstacles
        self.ub_u = ub_u;  self.lb_u = lb_u; self.ub_alpha = ub_alpha; self.lb_alpha = lb_alpha;

        [E, nu, Gc, ell, cw] = get_material_info(para);
        [case, p, q, w_type, delta, eta, kappa] = get_problem_constant_info(para);
        [mu, lambda_] = elas_para
        d = u.geometric_dimension()  # space dimension

        # set pde solver for displacement
        #--------------------------------------------
        v, du = TestFunction(V_pde), TrialFunction(V_pde);
        v_adj, du_adj = TestFunction(V_adj), TrialFunction(V_adj);
        del_alpha, dalpha = TestFunction(V_alpha), TrialFunction(V_alpha);

        self.pde = phi(self.alpha, p, delta)* inner(sigma_0(self.u, mu, lambda_, d), epsilon(v))*dx - dot(f_pde, v)*dx - dot(g_L1, v)*ds(1) - dot(g_L2, v)*ds(2) - dot(g_T3, v)*ds(3)- dot(g_T4, v)*ds(4) - dot(g_R5, v)*ds(5) - dot(g_R6, v)*ds(6) - dot(g_R7, v)*ds(7) - dot(g_B8, v)*ds(8) - dot(g_B9, v)*ds(9)

        self.pde_u = derivative(self.pde, self.u, du)
        self.problem_pde = NonlinearVariationalProblem(self.pde, self.u, bcs=bcs_pde, J=self.pde_u)
        self.problem_pde.set_bounds(lb_u.vector(), ub_u.vector())
        self.solver_pde = NonlinearVariationalSolver(self.problem_pde)
        self.solver_pde.parameters.update(para["solver_u"])

        # set adjoint solver for Lagrange multiplier
        #--------------------------------------------
        #NOTE: sigma_0(u_adj, mu, lambda_, d).n vanish on the Neumann boundary
        self.adj = phi(self.alpha, p, delta)* inner(sigma_0(self.u_adj, mu, lambda_, d), epsilon(v_adj))*dx + dot(f_pde, v_adj)*dx + dot(g_L1, v_adj)*ds(1) + dot(g_L2, v_adj)*ds(2) + dot(g_T3, v_adj)*ds(3) + dot(g_T4, v_adj)*ds(4) + dot(g_R5, v_adj)*ds(5) + dot(g_R6, v_adj)*ds(6) + dot(g_R7, v_adj)*ds(7) + dot(g_B8, v_adj)*ds(8) + dot(g_B9, v_adj)*ds(9)
        self.adj_u = derivative(self.adj, self.u_adj, du_adj)

        # Define the upper and lower bound for adj
        ub_adj_ex = Expression(("xmax","ymax"), degree=1, xmax=1.e6,  ymax=1.e6)
        lb_adj_ex = Expression(("xmin","ymin"), degree=1, xmin=-1.e6, ymin=-1.e6)
        self.lb_adj = interpolate(lb_adj_ex, V_adj);
        self.ub_adj = interpolate(ub_adj_ex, V_adj);

        self.problem_adj = NonlinearVariationalProblem(self.adj,self.u_adj, bcs=bcs_adj, J=self.adj_u)
        self.problem_adj.set_bounds(self.lb_adj.vector(), self.ub_adj.vector())
        self.solver_adj = NonlinearVariationalSolver(self.problem_adj)
        self.solver_adj.parameters.update(para["solver_u"])

        # set solver to find the gradient of the inner problem that help determine the active set
        #--------------------------------------------
        self.bcs_as = []
        # define Trial and Test Functions
        self.g_as, g_as_test = Function(V_adj), TestFunction(V_adj)

        # weak form for setting up the state equation
        self.F_as = inner(self.g_as, g_as_test)*dx - (phi(self.alpha, p, delta)* inner(sigma_0(self.u_adj, mu, lambda_, d), epsilon(v_adj))*dx + dot(f_pde, v_adj)*dx + dot(g_L1, v_adj)*ds(1) + dot(g_L2, v_adj)*ds(2) + dot(g_T3, v_adj)*ds(3) + dot(g_T4, v_adj)*ds(4) + dot(g_R5, v_adj)*ds(5) + dot(g_R6, v_adj)*ds(6) + dot(g_R7, v_adj)*ds(7) + dot(g_B8, v_adj)*ds(8) + dot(g_B9, v_adj)*ds(9))
        
        self.H_as = derivative(self.F_as, self.g_as, TrialFunction(V_adj))

        self.problem_as = NonlinearVariationalProblem(self.F_as, self.g_as, bcs= self.bcs_as, J= self.H_as)
        self.solver_as = NonlinearVariationalSolver(self.problem_as)
        
        
        # The Lagrange function
        #--------------------------------------------
        self.L = dot(f_pde, self.u - self.u_adj) * dx + eta * (self.alpha ** q) * dx + kappa*Gc/(2. * cw) * ( w(self.alpha, w_type)/ell + ell*inner(grad(self.alpha) , grad(self.alpha)))*dx + phi(self.alpha, p, delta)* inner(sigma_0(self.u, mu, lambda_, d), epsilon(self.u_adj))*dx + dot(g_L1, self.u -self.u_adj)*ds(1) + dot(g_L2, self.u -self.u_adj)*ds(2) + dot(g_T3, self.u -self.u_adj)*ds(3) + dot(g_T4, self.u -self.u_adj)*ds(4) + dot(g_R5, self.u -self.u_adj)*ds(5) + dot(g_R6, self.u -self.u_adj)*ds(6) + dot(g_R7, self.u -self.u_adj)*ds(7) + dot(g_B8, self.u -self.u_adj)*ds(8) + dot(g_B9, self.u -self.u_adj)*ds(9)
        # The first derivative of the Lagrange function w.r.t alpha
        #--------------------------------------------
        self.L_alpha = derivative(self.L, self.alpha, dalpha)

        # post processing parameters
        option = "TAO_pp" # see in the user_parameters
        [TAO_print_function_val, TAO_print_gradient_val, TAO_saving_frequency, TAO_save_pvd_u, TAO_save_pvd_u_adj, TAO_save_pvd_alpha, TAO_save_xdmf_u, TAO_save_xdmf_u_adj, TAO_save_xdmf_alpha] = get_post_processing_parameters(para, option)

        self.TAO_print_function_val = TAO_print_function_val
        self.TAO_print_gradient_val = TAO_print_gradient_val
        self.TAO_saving_frequency = TAO_saving_frequency
        self.TAO_save_pvd_u = TAO_save_pvd_u
        self.TAO_save_pvd_u_adj = TAO_save_pvd_u_adj
        self.TAO_save_pvd_alpha = TAO_save_pvd_alpha
        self.TAO_save_xdmf_u = TAO_save_xdmf_u
        self.TAO_save_xdmf_u_adj = TAO_save_xdmf_u_adj
        self.TAO_save_xdmf_alpha = TAO_save_xdmf_alpha

        self.xdmf_alpha = XDMFFile(mesh.mpi_comm(), savedir + "/TAO_blmvm/alpha.xdmf")
        self.vtk_alpha = File(savedir + "/TAO_blmvm/alpha.pvd")
        self.xdmf_u = XDMFFile(mesh.mpi_comm(), savedir + "/TAO_blmvm/u.xdmf")
        self.vtk_u = File(savedir + "/TAO_blmvm/u.pvd")
        self.xdmf_u_adj = XDMFFile(mesh.mpi_comm(), savedir + "/TAO_blmvm/u_adj.xdmf")
        self.vtk_u_adj = File(savedir + "/TAO_blmvm/u_adj.pvd")
        self.xdmf_active_set_lb = XDMFFile(mesh.mpi_comm(), savedir + "/TAO_blmvm/active_set_lb.xdmf")
        self.vtk_active_set_lb = File(savedir + "/TAO_blmvm/active_set_lb.pvd")
        self.xdmf_active_set_ub = XDMFFile(mesh.mpi_comm(), savedir + "/TAO_blmvm/active_set_ub.xdmf")
        self.vtk_active_set_ub = File(savedir + "/TAO_blmvm/active_set_ub.pvd")

        self.t = 0.1; self.count = 0;

    def f(self, x):
        self.alpha.vector()[:] = x
        self.solve_pde()
        self.solve_adj() # Old method without checking inner gradient
        #self.solve_adj_new() # New updated version when we check inner gradient to determine active set
        f_val = assemble(self.L)
        # Print variable info
        if self.TAO_print_function_val:
            alpha_max = self.alpha.vector().get_local().max()
            alpha_min = self.alpha.vector().get_local().min()
            alpha_norm2 = sqrt(assemble(inner(self.alpha, self.alpha)*dx))
            u_min = self.u.vector().get_local().min()
            u_norm2 = sqrt(assemble(inner(self.u, self.u)*dx))
            u_adj_norm2 = sqrt(assemble(inner(self.u_adj, self.u_adj)*dx))

            print("\n\tEvaluating the objective function for blmvm")
            print("\n\t||alpha||_l2 = %s;\n\talpha_max = %s;\talpha_min = %s  " % (alpha_norm2, alpha_max, alpha_min))
            print("\n\tu_min = %f; ||u||_l2 = %f; ||u_adj||_l2 = %f\n" % (u_min, u_norm2, u_adj_norm2))
        #Saving xdmf files
        #--------------------------------------------
        self.count +=1;
        if self.count % self.TAO_saving_frequency == 0:
            if self.TAO_save_xdmf_alpha:
                with self.xdmf_alpha as file_alpha:
                    file_alpha.write_checkpoint(self.alpha, "alpha")
            if self.TAO_save_pvd_alpha:
                self.vtk_alpha << (self.alpha,self.t)

            if self.TAO_save_xdmf_u:
                with self.xdmf_u as file_u:
                    file_u.write_checkpoint(self.u, "u")
            if self.TAO_save_pvd_u:
                self.vtk_u << (self.u,self.t)

            if self.TAO_save_xdmf_u_adj:
                with self.xdmf_u_adj as file_u_adj:
                    file_u_adj.write_checkpoint(self.u_adj, "u_adj")
            if self.TAO_save_pvd_u_adj:
                self.vtk_u_adj << (self.u_adj,self.t)
                
            #save the active set
            with self.xdmf_active_set_lb as file_active_set_lb:
                file_active_set_lb.write_checkpoint(self.active_set_lb, "active_set_lb")
            self.vtk_active_set_lb << (self.active_set_lb,self.t)
            
            with self.xdmf_active_set_ub as file_active_set_ub:
                file_active_set_ub.write_checkpoint(self.active_set_ub, "active_set_ub")
            self.vtk_active_set_ub << (self.active_set_ub,self.t)
            self.t +=0.1
        return f_val

    def F(self, b, x):
        self.alpha.vector()[:] = x
        #self.solve_pde() # No need to recompute u this here
        #self.solve_adj() # No need to recompute u_adj this here
        assemble(self.L_alpha, tensor=b)

        # Print varialbe info
        if self.TAO_print_gradient_val:
            u_norm2 = sqrt(assemble(inner(self.u, self.u)*dx))
            u_adj_norm2 = sqrt(assemble(inner(self.u_adj, self.u_adj)*dx))
            print("\n\tEvaluating the gradient of the objective function for blmvm")
            print("\n\t||u||_l2 = %f; ||u_adj||_l2 = %f\n" % (u_norm2, u_adj_norm2))
            return

    def J(self, A, x):
        pass

    def get_active_set(self):
        u_array = self.u.vector().get_local();
        ub_u_array = self.ub_u.vector().get_local();
        lb_u_array = self.lb_u.vector().get_local();
        # Find the active set and save it in a list, UAS
        self.UAS = list(u_array[:] >(ub_u_array[:] - 1e-8))
        self.LAS = list(u_array[:] <(lb_u_array[:] + 1e-8))
        
        #Set upper and lower bounds for adjoint
        self.active_set_lb.vector()[:] = self.LAS;
        self.active_set_ub.vector()[:] = self.UAS;
        return

    def set_bounds_adj(self):
        self.ub_adj.vector()[:] = 1e6
        self.lb_adj.vector()[:] = -1e6
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


    def get_active_set_and_set_adj_bounds(self):
        #self.problem_as = NonlinearVariationalProblem(self.F_as, self.g_as, bcs=self.bcs_as, J= self.H_as)
        #self.solver_as = NonlinearVariationalSolver(self.problem_as)
        #self.solver_as.solve() #compute self.g_as
        
        # get array of the gradient of the inner problem
        #g_as_array = self.g_as.vector().get_local();

        # get state arrays
        u_array = self.u.vector().get_local();
        ub_u_array = self.ub_u.vector().get_local();
        lb_u_array = self.lb_u.vector().get_local();

        # get active set arrays
        LAS_array = self.active_set_lb.vector().get_local()
        UAS_array = self.active_set_ub.vector().get_local()

        # Find the active set and save it in a list, UAS
        c = 1e6;
            
        #LAS_array[:] = g_as_array[:] + c*(lb_u_array[:] - u_array[:])
        #UAS_array[:] = g_as_array[:] + c*(ub_u_array[:] - u_array[:])
        
        LAS_array[:] = (lb_u_array[:] - u_array[:])
        UAS_array[:] = (ub_u_array[:] - u_array[:])
                    
        LAS = list(LAS_array[:] >= 0.)
        UAS = list(UAS_array[:] <= 0.)
            
        #Set upper and lower bounds for adjoint
        self.active_set_lb.vector()[:] = LAS;
        self.active_set_ub.vector()[:] = UAS;
            
            
        self.ub_adj.vector()[:] = 1e6
        self.lb_adj.vector()[:] = -1e6
            
        ub_adj_array = self.ub_adj.vector().get_local();
        lb_adj_array = self.lb_adj.vector().get_local();
                    
                    
        #Set upper and lower bounds for adjoint
        for at_dof, is_active_point in enumerate(UAS): # for UAS
            if is_active_point == 1 :
                ub_adj_array[at_dof] = 0.; lb_adj_array[at_dof] = 0.;

        for at_dof, is_active_point in enumerate(LAS): # for LAS
            if is_active_point == 1 :
                ub_adj_array[at_dof] = 0.; lb_adj_array[at_dof] = 0.;
                            
        self.ub_adj.vector()[:] = ub_adj_array;
        self.lb_adj.vector()[:] = lb_adj_array;
        return


    def solve_pde(self):
        self.problem_pde = NonlinearVariationalProblem(self.pde, self.u, bcs=self.bcs_pde, J=self.pde_u)
        self.problem_pde.set_bounds(self.lb_u.vector(), self.ub_u.vector())
        self.solver_pde = NonlinearVariationalSolver(self.problem_pde)
        self.solver_pde.parameters.update(self.para["solver_u"])
        self.solver_pde.solve()
        return


    def solve_adj(self):
        self.problem_adj = NonlinearVariationalProblem(self.adj, self.u_adj, bcs=self.bcs_adj, J=self.adj_u)
        self.solver_adj = NonlinearVariationalSolver(self.problem_adj)
        self.solver_adj.parameters.update(self.para["solver_u"])
        self.get_active_set()
        self.set_bounds_adj()
        self.solver_adj.solve()


    def solve_adj_new(self):
        self.problem_adj = NonlinearVariationalProblem(self.adj, self.u_adj, bcs=self.bcs_adj, J=self.adj_u)
        self.solver_adj = NonlinearVariationalSolver(self.problem_adj)
        self.solver_adj.parameters.update(self.para["solver_u"])
        self.get_active_set_and_set_adj_bounds()
        self.problem_adj.set_bounds(self.lb_adj.vector(), self.ub_adj.vector())
        self.solver_adj.solve()

        return


# Define strain and stress
def epsilon(u):
    return sym(nabla_grad(u))

def sigma_0(u, mu, lambda_, d):
    return lambda_*tr(epsilon(u))*Identity(d) + 2.*mu*epsilon(u)

# Constitutive functions of the damage model
def w(alpha, w_type):
    if w_type == 1:
        return alpha*(1-alpha)
    elif w_type ==2:
        return alpha**2. * (1-alpha)**2.
    elif w_type ==3:
        return alpha*(1-alpha**2.)
    else:
        print("\n Please consider w_type again")
        exit()

# The characteristic function
def phi(alpha, p, delta):
    return (1. - delta)*alpha**p + delta


def alpha_initial_guess(V_alpha, para, Lx, Ly):
    [is_constant, alpha_val, m, n] = get_initial_guess_info(para);
    if is_constant:
        alpha_0 = Expression('alpha_val', degree=1, alpha_val = alpha_val)
    else :
        alpha_0 = Expression('0.5*alpha_val + 0.5*sin(m*pi*x[0]/Lx)*sin(n*pi*x[1]/Ly)', degree=1, alpha_val = alpha_val, m = m, n = n, Lx = Lx, Ly = Ly)
    return alpha_0



def boundary_condition(V_pde, V_adj, V_alpha, sub_domains, example):
    
    [sub_left_1, sub_left_2, sub_top_3, sub_top_4, sub_right_5, sub_right_6, sub_right_7, sub_bottom_8, sub_bottom_9] = sub_domains
    
    # Dirichlet B.C. for displacement field
    bcl_1_pde = DirichletBC(V_pde, Constant((0., 0.)), sub_left_1);
    bcl_2_pde = DirichletBC(V_pde, Constant((0., 0.)), sub_left_2);
    bcr_5_pde = DirichletBC(V_pde.sub(0), Constant(0.), sub_right_5);
    bcr_6_pde = DirichletBC(V_pde.sub(0), Constant(0.), sub_right_6);
    bcr_7_pde = DirichletBC(V_pde.sub(0), Constant(0.), sub_right_7);
    bcs_pde = [bcl_1_pde, bcl_2_pde, bcr_5_pde, bcr_6_pde, bcr_7_pde]
    
    # Dirichlet B.C. for Lagrange multiplier
    bcl_1_adj = DirichletBC(V_adj, Constant((0., 0.)), sub_left_1);
    bcl_2_adj = DirichletBC(V_adj, Constant((0., 0.)), sub_left_2);
    bcr_5_adj = DirichletBC(V_adj.sub(0), Constant(0.), sub_right_5);
    bcr_6_adj = DirichletBC(V_adj.sub(0), Constant(0.), sub_right_6);
    bcr_7_adj = DirichletBC(V_adj.sub(0), Constant(0.), sub_right_7);
    bcs_adj = [bcl_1_adj, bcl_2_adj, bcr_5_adj, bcr_6_adj, bcr_7_adj]
    # Note 1: The displacement is fixed in the x-direction on the right boundary and clamped on the left boundary.
    # Note 2: In general, V_pde != V_adj
    # B.C. for alpha
    bcs_alpha = []
    return bcs_pde, bcs_adj, bcs_alpha


def create_mesh(para, savedir):
    
    # Get mesh info
    [cell_size, Lx, Ly, structured_mesh, resolution] = get_mesh_info(para);
    Nx = int(Lx/cell_size);Ny = int(Ly/cell_size);
    case = para["problem"]["case"];
    example = para["ex"]
        
    parameters["ghost_mode"] = "shared_facet" #None,shared_vertex, shared_facet
    parameters["reorder_cells_gps"] = True
    parameters["reorder_vertices_gps"] = True
    parameters["dof_ordering_library"] = "SCOTCH"
    parameters["mesh_partitioner"] = "SCOTCH" #SCOTCH, ParMETIS(donot support), METIS(unkown), Chaco and CHACO(Unkown)
    parameters["partitioning_approach"] = "PARTITION" #REPARTITION, PARTITION, REFINE
    File(savedir + "/mesh_partition_parameters%s_%s.xml"%(example, case)) << parameters

    geom=Rectangle(Point(0., 0.), Point(Lx, Ly))
    if structured_mesh:
        mesh = RectangleMesh(Point(0.,0.), Point(Lx, Ly), Nx, Ny, "crossed")
    else:
        mesh = generate_mesh(geom, resolution)
    
    # Define the boundaries
    class Left_2(SubDomain): #x = 0.; Ly/10 < y < Ly
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0., DOLFIN_EPS) and near(x[1], Ly, 0.9*Ly + 2*cell_size)

    class Left_1(SubDomain): #x = 0.;  0 < y < Ly/10
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0., DOLFIN_EPS) and near(x[1], 0., 0.1*Ly + DOLFIN_EPS)
    
    class Top_3(SubDomain): #0 < x < 5/6*Lx; y = Ly;
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Ly, DOLFIN_EPS) and near(x[0], 0., 5./6.*Lx + 2*cell_size)
    
    class Top_4(SubDomain): # 5/6*Lx < x < Lx; y = Ly;
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Ly, DOLFIN_EPS) and near(x[0], Lx, Lx/6. + DOLFIN_EPS)
    
    class Right_5(SubDomain): #x = Lx; 2/3*Ly < y < Ly
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], Lx, DOLFIN_EPS) and near(x[1], Ly, Ly/3. + 2*cell_size)

    class Right_7(SubDomain): #x = Lx; 0 < y < 1/3*Ly
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], Lx, DOLFIN_EPS) and near(x[1], 0., Ly/3. + 2*cell_size)

    class Right_6(SubDomain): #x = Lx; 1/3*Ly < y < 2/3*Ly
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], Lx, DOLFIN_EPS) and near(x[1], Ly/2., Ly/6.)

    class Bottom_8(SubDomain): #9Lx/10 < x < Lx; y = 0
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0., DOLFIN_EPS) and near(x[0], Lx, 0.9*Lx + 2*cell_size)

    class Bottom_9(SubDomain): #0 < x < Lx/10; y = 0
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0., DOLFIN_EPS) and near(x[0], 0., 0.1*Lx)

    sub_left_1 = Left_1(); sub_left_2 = Left_2();
    sub_top_3 = Top_3(); sub_top_4 = Top_4();
    sub_right_5 = Right_5(); sub_right_6 = Right_6(); sub_right_7 = Right_7();
    sub_bottom_8 = Bottom_8(); sub_bottom_9 = Bottom_9();
    
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0) # marked as 0
    sub_left_2.mark(boundary_markers, 2) # marked as 2
    sub_left_1.mark(boundary_markers, 1) # marked as 1
    sub_top_3.mark(boundary_markers, 3) # marked as 3
    sub_top_4.mark(boundary_markers, 4) # marked as 4
    sub_right_5.mark(boundary_markers, 5) # marked as 5
    sub_right_7.mark(boundary_markers, 7) # marked as 7
    sub_right_6.mark(boundary_markers, 6) # marked as 6
    sub_bottom_8.mark(boundary_markers, 8) # marked as 8
    sub_bottom_9.mark(boundary_markers, 9) # marked as 9
    
    sub_domains = [sub_left_1, sub_left_2, sub_top_3, sub_top_4, sub_right_5, sub_right_6, sub_right_7, sub_bottom_8, sub_bottom_9]
    
    # Save boundary markers
    File(savedir + "/mesh%s/boundary_markers%s.pvd"%(case, case)) << boundary_markers
    # Redefine boundary integration measure .NOTE: ds is for external bounray. dS for internal boundary
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    the_domain = [mesh, boundary_markers, sub_domains, ds]
    
    # Create MeshFunction to hold cell process rank
    partitions = MeshFunction('size_t', mesh, mesh.topology().dim(), MPI.rank(mesh.mpi_comm()))

    # Save colored mesh partitions in VTK format if running in parallel

    File(savedir + "/mesh%s/mesh_partitions_%s.pvd"%(case, case))<<partitions
    
    return the_domain


def get_forces(example, para):
    
    f_val = para["problem"]["force"]["f"];
    gx = para["problem"]["force"]["gx"];
    gy = para["problem"]["force"]["gy"];
    
    # Body force
    f = Constant((0., -f_val));
    
    # Traction forces
    if example == 1:
        # traction force is applied on the top-right boundary
        g_L1 = Constant((0., 0.)); g_L2 = Constant((0., 0.));
        g_T3 = Constant((0., 0.)); g_T4 = Constant((gx, gy));
        g_R5 = Constant((0., 0.)); g_R6 = Constant((0., 0.)); g_R7 = Constant((0., 0.));
        g_B8 = Constant((0., 0.)); g_B9 = Constant((0., 0.));
        print("Traction forces: g_L1 = (0, 0); g_L2 = (0, 0); g_T3 = (0, 0); g_T4 = ({:.3f}, {:.3f}); g_R5 = (0, 0); g_R6 = (0, 0); g_R7 = (0, 0); g_B8 = (0, 0); g_B9 = (0, 0)" .format(gx, gy))
    elif example == 3:
        # traction force is applied on the right-center boundary
        g_L1 = Constant((0., 0.)); g_L2 = Constant((0., 0.));
        g_T3 = Constant((0., 0.)); g_T4 = Constant((0., 0.));
        g_R5 = Constant((0., 0.)); g_R6 = Constant((gx, gy)); g_R7 = Constant((0., 0.));
        g_B8 = Constant((0., 0.)); g_B9 = Constant((0., 0.));
        print("Traction forces: g_L1 = (0, 0); g_L2 = (0, 0); g_T3 = (0, 0); g_T4 = (0, 0); g_R5 = (0, 0); g_R6 = ({:.3f}, {:.3f}); g_R7 = (0, 0); g_B8 = (0, 0);  g_B9 = (0, 0)" .format(gx, gy))
    else:
        # traction force is applied on the top boundary
        g_L1 = Constant((0., 0.)); g_L2 = Constant((0., 0.));
        g_T3 = Constant((gx, gy)); g_T4 = Constant((gx, gy));
        g_R5 = Constant((0., 0.)); g_R6 = Constant((0., 0.)); g_R7 = Constant((0., 0.));
        g_B8 = Constant((0., 0.)); g_B9 = Constant((0., 0.));
        print("Traction forces: g_L1 = (0, 0); g_L2 = (0, 0); g_T3 = ({:.3f}, {:.3f}); g_T4 = ({:.3f}, {:.3f}); g_R5 = (0, 0); g_R6 = (0, 0); g_R7 = (0, 0); g_B8 = (0, 0);  g_B9 = (0, 0)" .format(gx, gy, gx, gy))
    
    the_forces = [f, g_L1, g_L2, g_T3, g_T4, g_R5, g_R6, g_R7, g_B8, g_B9];
    return the_forces

def get_obstacle_constraints(example, para, V_pde, V_alpha, savedir):
    
    [ub_u_x, ub_u_y, lb_u_x, lb_u_y, ub_alpha, lb_alpha] = get_problem_obstacle_info(para)
    Lx = para["mesh"]["Lx"]; Ly = para["mesh"]["Ly"];
    case = para["problem"]["case"];
    #-----------------------------------------------------------------
    # Set constraints for u
    #-----------------------------------------------------------------
    if example == 3:
        # Two obstacles case. Define my functions for constraints
        #------------------------------------------
        class my_ub(UserExpression):
            def __init__(self, ub_u_y, x1, x2, Lx, Ly, **args):
                self.ub_u_y = ub_u_y; self.x1 = x1; self.x2 = x2
                self.Lx = Lx; self.Ly = Ly;
                super().__init__(**args)
            def eval_cell(self, values, x, cell):
                if near(x[1], self.Ly, DOLFIN_EPS) and (self.x1 < x[0]) and (x[0] < self.x2):
                    values[0] = 10.; values[1] = self.ub_u_y - x[1];
                else:
                    values[0] = 10.; values[1] = 10.; # infty := 10
            def value_shape(self):
                return (2,)
        
        class my_lb(UserExpression):
            def __init__(self, lb_u_y, x1, x2, Lx, Ly, **args):
                self.lb_u_y = lb_u_y; self.x1 = x1; self.x2 = x2
                self.Lx = Lx; self.Ly = Ly;
                super().__init__(**args)
            def eval_cell(self, values, x, cell):
                if near(x[1], 0., DOLFIN_EPS) and (self.x1 < x[0]) and (x[0] < self.x2):
                    values[0] = -10.; values[1] = self.lb_u_y - x[1];
                else:
                    values[0] = -10.; values[1] = -10.;
            def value_shape(self):
                return (2,)
        #------------------------------------------
        ub_Ex = my_ub( ub_u_y, 0.1*Lx, 0.3*Lx, Lx, Ly, degree=1)
        lb_Ex = my_lb( lb_u_y, 0.4*Lx, 0.6*Lx, Lx, Ly, degree=1)
        lb_u = interpolate(lb_Ex, V_pde); ub_u = interpolate(ub_Ex, V_pde)
    
    else:
        # without or with one obstacle
        ub_Ex = Expression(("xmax","ymax"), degree=1, xmax=ub_u_x, ymax=ub_u_y)
        lb_Ex = Expression(("xmin","ymin - x[1]"), degree=1, xmin=lb_u_x, ymin=lb_u_y)
        lb_u = interpolate(lb_Ex, V_pde); ub_u = interpolate(ub_Ex, V_pde)
    
    #-----------------------------------------------------------------
    # Set constraints for alpha
    #-----------------------------------------------------------------
    if example == 2:
        class lb_alpha_Ex(UserExpression):
            def __init__(self, lb_alpha, ub_alpha, Lx, Ly, **args):
                self.lb_alpha = lb_alpha; self.ub_alpha = ub_alpha;
                self.Lx = Lx; self.Ly = Ly;
                super().__init__(**args)
            def eval_cell(self, values, x, cell):
                if near(x[1], self.Ly, DOLFIN_EPS):
                    values[0] = self.ub_alpha;
                else:
                    values[0] =  self.lb_alpha;
        #------------------------------------------
        lb_alpha_Ex2 = lb_alpha_Ex(lb_alpha, ub_alpha, Lx, Ly, degree=1)
        
        ub_alpha = interpolate(Constant(ub_alpha), V_alpha)
        lb_alpha = interpolate(lb_alpha_Ex2, V_alpha)
    else:
        ub_alpha = interpolate(Constant(ub_alpha), V_alpha)
        lb_alpha = interpolate(Constant(lb_alpha), V_alpha)

    # Save in pvd files
    File(savedir + "/obstacles%s/ub_u%s.pvd"%(case, case))<< ub_u
    File(savedir + "/obstacles%s/lb_u%s.pvd"%(case, case))<< lb_u
    File(savedir + "/obstacles%s/ub_alpha%s.pvd"%(case, case))<< ub_alpha
    File(savedir + "/obstacles%s/lb_alpha%s.pvd"%(case, case))<< lb_alpha
    # Save in XDMF files
    #    XDMFFile(savedir + "/obstacles%s/ub_u%s.xdmf"%(case, case)).write(ub_u)
    #    XDMFFile(savedir + "/obstacles%s/lb_u%s.xdmf"%(case, case)).write(lb_u)
    #    XDMFFile(savedir + "/obstacles%s/ub_alpha%s.xdmf"%(case, case)).write(ub_alpha)
    #    XDMFFile(savedir + "/obstacles%s/lb_alpha%s.xdmf"%(case, case)).write(lb_alpha)
    return ub_u, lb_u, ub_alpha, lb_alpha


def save_opt_sol(alpha_opt, u_opt, u_adj_opt, active_set_lb_opt, active_set_ub_opt, para, savedir, mesh):
    option = "opt_pp" # opt_post_processing_parameters
    case = para["problem"]["case"];
    
    [save_opt_u, save_opt_active_set_ub, save_opt_active_set_lb, save_opt_u_adj, save_opt_alpha, save_opt_stress] = get_post_processing_parameters(para, option)
    
    if save_opt_u:
        File(savedir + "/opt_sol%s/u_opt%s.pvd"%(case, case))<< u_opt;
        xdmf_u_opt = XDMFFile(mesh.mpi_comm(), savedir + "/opt_sol%s/u_opt%s.xdmf"%(case, case))
        with xdmf_u_opt as file_u_opt:
            file_u_opt.write_checkpoint(u_opt, "u_opt")
            
    if save_opt_active_set_ub:
        File(savedir + "/opt_sol%s/active_set_ub_opt%s.pvd"%(case, case))<< active_set_ub_opt;
        xdmf_active_set_ub_opt = XDMFFile(mesh.mpi_comm(), savedir + "/opt_sol%s/active_set_ub_opt%s.xdmf"%(case, case))
        with xdmf_active_set_ub_opt as file_active_set_ub_opt:
            file_active_set_ub_opt.write_checkpoint(active_set_ub_opt, "active_set_ub_opt")

            
    if save_opt_active_set_lb:
        File(savedir + "/opt_sol%s/active_set_lb_opt%s.pvd"%(case, case))<< active_set_lb_opt;
        xdmf_active_set_lb_opt = XDMFFile(mesh.mpi_comm(), savedir + "/opt_sol%s/active_set_lb_opt%s.xdmf"%(case, case))
        with xdmf_active_set_lb_opt as file_active_set_lb_opt:
            file_active_set_lb_opt.write_checkpoint(active_set_lb_opt, "active_set_lb_opt")
            
    if save_opt_u_adj:
        File(savedir + "/opt_sol%s/u_adj_opt%s.pvd"%(case, case))<< u_adj_opt;
        xdmf_u_adj_opt = XDMFFile(mesh.mpi_comm(), savedir + "/opt_sol%s/u_adj_opt%s.xdmf"%(case, case))
        with xdmf_u_adj_opt as file_u_adj_opt:
            file_u_adj_opt.write_checkpoint(u_adj_opt, "u_adj_opt")
        
    if save_opt_alpha:
        File(savedir + "/opt_sol%s/alpha_opt%s.pvd"%(case, case))<< alpha_opt;
        xdmf_alpha_opt = XDMFFile(mesh.mpi_comm(), savedir + "/opt_sol%s/alpha_opt%s.xdmf"%(case, case))
        with xdmf_alpha_opt as file_alpha_opt:
            file_alpha_opt.write_checkpoint(alpha_opt, "alpha_opt")
    
    return


if __name__== "__main__":
    start = time. time()
    main()
    end = time. time()
    print("\nExecution time (in seconds):",(end - start))
