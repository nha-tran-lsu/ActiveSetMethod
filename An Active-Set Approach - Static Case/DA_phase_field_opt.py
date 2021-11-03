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
    FEniCS program: Dolfin - Adjoint
    """


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. py:currentmodule:: dolfin_adjoint
# This is a modification of the
# Mathematical Programs with Equilibrium Constraints
# ==================================================
# written by Simon W. Funke
# .. sectionauthor:: Simon W. Funke <simon@simula.no>
#
# This demo solves example 5.2 of :cite:`hintermueller2011`.
#
# Problem definition
# ******************
#
# This problem is to minimise
#
# .. math::
#       \min_{J(alpha, u(alpha)) =  \int_{\Omega} f u dx + \int_{\parttial\Omega} g u ds + \eta \int_{\Omega} \alpha^q dx} + \kappa \frac{Gc}{c_w} \int_\Omega \frac{w(alpha)}{\ell} + \ell \nabla(\alpha)\cdot\nabla(\alpha) dx
#
# subject to the variational inequality
#
# .. math::
#       ( \Phi(\alpha) A e(u), e(v - u) )_\Omega &\ge (f, v - u)_\Omega + (g, v - u)_{\partial\Omega} \qquad \forall v \ge 0, v = 0 \ \mathrm{on}\ \patial_D \Omega, \\
#       u &\ge u_min, \quad u &leq u_max \\
#       u &= 0 \quad \mathrm{on}\ \partial_D \Omega,

# and control constraints
#
# .. math::
#          0 \le \alpha \le 1 \qquad \forall x \in \Omega,
#
#
# where :math:`alpha` is the control, :math:`u` is the solution of the
# variational inequality, :math:`f`is body force, :math:`g` is traction force
# and :math:`0, 1` are upper and lower bounds for the
# control.
#
# This problem is fundamentally different to a PDE-constrained
# optimisation problem in that the constraint is not a PDE, but a
# variational inequality.  Such problems are called Mathematical
# Programs with Equilibrium Constraints (MPECs) and have applications
# in engineering design (e.g. to determine optimal trajectories for
# robots :cite:`yunt2005` or process optimisation in chemical
# engineering :cite:`baumrucker2008`) and in economics (e.g. in
# leader-follower games :cite:`leyffer2005` and optimal pricing
# :cite:`lawphongpanich2004`).
#
# Even though it is known that the above problem admits a unique
# solution, there are some difficulties to be considered when solving
# MPECs:
#
#  - the set of feasible points is in general not necessarly convex or connected, and
#  - the reduced problem is not Fréchet-differentiable.
#
# Following :cite:`hintermueller2011`, we will overcome these issues
# in the next section with a penalisation approach.  For a more
# thorough discussion on MPECs, see :cite:`luo1996` and the references
# therein.
#
# Penalisation technique
# **********************
#
# A common approach for solving variational inequalities is to
# approximate them by a sequence of nonlinear PDEs with a penalisation
# term.  We transform the above problem into a sequence of
# PDE-constrained optimisation problems, which can be solved with
# ``dolfin-adjoint``.
#
# For the above problem we use the approximation
#
# .. math::
#       ( \Phi(\alpha) A e(u), e(v - u) )_\Omega + \frac{1}{\rho} (\pi(u), v)_\Omega = (f + u, v)_\Omega + (g, v - u)_{\partial\Omega} \forall v \ge 0, v = 0 \ \mathrm{on}\ \patial_D \Omega, \\
#
# where :math:`\rho > 0` is the penalty parameter and the penalty term
# is defined as
#
# .. math::
#       \pi(u) = -\max(0, u).
#
#
# This approximation yields solutions which converge to the solution of
# the variational inequality as :math:`\alpha \to 0` (see chapter IV of
# :cite:`kinderlehrer2000`).
#
# In order to be able to apply a gradient-based optimisation method, we
# need differentiabilty of the above equation.  The :math:`\max`
# operator is not differentiable at the origin, and hence it is replaced
# by a smooth (:math:`C^1`) approximation (plot modified from
# :cite:`hintermueller2011`):
#
# .. math::
#       {\max}_{\epsilon}(0, y) =
#       \begin{cases}
#       y - \frac{\epsilon}{2} & \mbox{if } y \ge \epsilon, \\
#                     \frac{y^2}{2\epsilon}  & \mbox{if } y \in (0, \epsilon), \\
#                     0                  & \mbox{if } y \le 0.
#       \end{cases}
#
#
# .. image:: mpec-smoothmax.jpg
#     :scale: 50
#     :align: center
#
#
# Implementation
# **************
#
# First, the :py:mod:`dolfin` and :py:mod:`dolfin_adjoint` modules are
# imported. We also tell DOLFIN to only print error messages to keep the
# output comprehensible:
from user_parameters import * #
from dolfin import *
from dolfin_adjoint import *
from ufl.operators import Max
from mshr import *
import scipy
print ("The version of scipy is ", scipy.__version__)
import numpy as np
import sys, os, sympy, math, time
import petsc4py as PETSc
import matplotlib.pyplot as plt
from itertools import count

set_log_level(LogLevel.ERROR)#NOTE: Comment this line to monitor the PDE solver at each iteration of optimization process

def main():
    parameters["form_compiler"]["cpp_optimize"] = True
    # Needed to have a nested conditional
    parameters["form_compiler"]["representation"] = "uflacs"
    mpi_comm = MPI.comm_world; my_rank = MPI.rank(mpi_comm)
    pinfty = 1.e8; ninfty = -1.e8;
    
    # Pick an example:
    # 1. no_obstacle , 2. one_obstacle (default), or 3. two_obstacles
    para = default_parameters(); # load the default parameters
    para.add("gtol", 5e-8)
    para.parse(); # take inputs from the command line
    example = para["ex"]
    
    gtol = para["gtol"]
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
        prefix = "DA_no_obstacle" + "/DA_%s%s_"%(example, case)
    elif example == 3:
        prefix = "DA_two_obstacles" + "/DA_%s%s_"%(example, case)
    else:
        prefix = "DA_one_obstacle" + "/DA_%s%s_"%(example, case)

    savedir = "DA_results_2021_del_%s/%s_cell_size%s_Nx%s_Ny%s_Lx%s_Ly%sp%s_q%s_"%(delta, prefix, cell_size, Nx, Ny, Lx, Ly, p, q)
    File(savedir + "/DA_parameters_%s%s" % (example, case) +".xml") << para

    # Create mesh
    [mesh, boundary_markers, sub_domains, ds] = create_mesh(para, savedir);
    the_domain = [mesh, boundary_markers, sub_domains, ds]
    meshname = savedir + "/mesh/DA_mesh_%s%s"% (example, case) + ".xdmf"
    XDMFFile(MPI.comm_world, meshname).write(mesh)
    hmax = mesh.hmax(); hmin = mesh.hmin();
    num_cells = mesh.num_cells(); num_vertices = mesh.num_vertices();
    
    # Print parameters
    print("\n The parameters: \n", para.str(False));
    print("\n hmax = %f; hmin = %f; \nnum_cells = %f; num_vertices = %f" %(hmax, hmin, num_cells, num_vertices));
    print("Parameters: \n cell_size = {:.3f}; Nx = {:d}; Ny = {:d}; Lx = {:.1f}; Ly = {:.1f}; q = {:.1f}; p = {:.1f}; eta = {:.5f}; kappa = {:.15f}; Gc = {:.3f}; ell = {:.5f}; E = {:.1f}; nu = {:.1f}; delta = {:.10f}; w_type = {:d}, gtol = {:.8f}".format(cell_size, Nx, Ny, Lx, Ly, q, p, eta, kappa, Gc, ell, E, nu, delta, w_type, gtol))
    # Define function spaces (FS)
    V_pde = VectorFunctionSpace(mesh, 'P', 2) # Displacement FS
    V_alpha = FunctionSpace(mesh, 'P', 1) # Alpha FS

    # Boundary conditions
    bcs_pde = boundary_condition(V_pde, sub_domains, example);
    # Get forces
    [f, g_L1, g_L2, g_T3, g_T4, g_R5, g_R6, g_R7, g_B8, g_B9] = get_forces(example, para);
    forces = [f, g_L1, g_L2, g_T3, g_T4, g_R5, g_R6, g_R7, g_B8, g_B9]

    # Define solution variables
    u = Function(V_pde, name="Solution")
    alpha = Function(V_alpha, name="Control");
    alpha_2write = Function(V_alpha)
    v = TestFunction(V_pde)

    d = u.geometric_dimension()  # Space dimension
    print("dim = ", d)
    ic = Control(u) # Control variable


    # Next, we define and solve the variational formulation of the PDE
    # constraint with the penalisation parameter set to
    # :math:`\rho=10^{-4}`.  This initial value of :math:`\rho` will
    # then be iteratively reduced to better approximate the underlying MPEC.
    # To ensure that new values of :math:`\rho` are reflected on the tape,
    # we define a ``Placeholder`` before using it.

    rho = Constant(1e-4)
    from pyadjoint.placeholder import Placeholder
    Placeholder(rho)


    # Initial guess for alpha
    alpha_0 = alpha_initial_guess(V_alpha, para, Lx, Ly);
    alpha.assign(interpolate(alpha_0, V_alpha))

    # Get obstacle constraints
    ub_u, lb_u, ub_alpha, lb_alpha = get_obstacle_constraints(example, para, V_pde, V_alpha, savedir)
    obstacles = [ub_u, lb_u, ub_alpha, lb_alpha]

    # Definte the penalty term
    ux, uy = u.split(True)
    vx, vy = TestFunction(V_pde)
    penaltyTerms = inner(smoothmax(u[1] - ub_u[1]), vy)*dx - inner(smoothmax(lb_u[1] - u[1]), vy)*dx + inner(smoothmax(u[0] - ub_u[0]), vx)*dx - inner(smoothmax(lb_u[0] - u[0]), vx)*dx

    F = phi(alpha, p, delta) * inner(sigma_0(u, mu, lambda_, d), epsilon(v))*dx - dot(f, v)*dx - dot(g_L1, v)*ds(1) - dot(g_L2, v)*ds(2) - dot(g_T3, v)*ds(3) - dot(g_T4, v)*ds(4)- dot(g_R5, v)*ds(5)- dot(g_R6, v)*ds(6)- dot(g_R7, v)*ds(7) - dot(g_B8, v)*ds(8) - dot(g_B9, v)*ds(9) + 1./rho*penaltyTerms #NOTE: Taking into account the bounds for the state variable, u.

    solve(F == 0, u, bcs=bcs_pde)
    ## NOTE: we dont have to create a snes solver. just use "solve" as in the demo code for MPECs. This showed a better result for the toy problem with poisson equation constraint.
    ## We create a NonlinearVariationalSolver.
    #F_u = derivative(F, u)
    #problem = NonlinearVariationalProblem(F, u, bcs=bcs_pde, J=F_u)
    #solver = NonlinearVariationalSolver(problem)
    #snes_solver_parameters_bounds = {"nonlinear_solver": "snes", "snes_solver": {"linear_solver": "lu", "maximum_iterations": 50, "report": False,                "line_search": "basic","method":"default", "absolute_tolerance":1e-10, "relative_tolerance":1e-10, "solution_tolerance":1e-10}}
    ##NOTE: "The solver parameters are not "remembered" for Non-/LinearVariationalSolver." These parameters are remembered only once when we first run the code
    #solver.parameters.update(snes_solver_parameters_bounds)
    #(iter, converged) = solver.solve()
    #if not converged:
    #    warning("Convergence is not guaranteed when modifying some parameters or using PETSc.")


    # With the forward problem solved once, :py:mod:`dolfin_adjoint` has
    # built a *tape* of the forward model; it will use this tape to drive
    # the optimisation, by repeatedly solving the forward model and the
    # adjoint model for varying control inputs.
    #
    # We finish the initialisation part by defining the functional of
    # interest, the optimisation parameter and creating the :doc:`reduced
    # functional <../maths/2-problem>` object:

    J = assemble(dot(f, u) * dx + dot(g_L1, u)*ds(1) + dot(g_L2, u)*ds(2) + dot(g_T3, u)*ds(3) + dot(g_T4, u)*ds(4) + dot(g_R5, u)*ds(5)+ dot(g_R6, u)*ds(6)+ dot(g_R7, u)*ds(7) + dot(g_B8, u)*ds(8) + dot(g_B9, u)*ds(9) + eta* alpha**q *dx + Gc * kappa/(2.*cw) * (w(alpha, w_type)/ell + ell * inner(grad(alpha) , grad(alpha)))*dx)
    
    # Formulate the reduced problem
    m = Control(alpha)  # Create a parameter from u, as it is the variable we want to optimise
    Jhat = ReducedFunctional(J, m)

    # Create output files
    u_pvd = File(savedir + "/opt_sol%s/u_opt_%s"%(case, case) + ".pvd")
    alpha_pvd = File(savedir + "/opt_sol%s/alpha_opt_%s"%(case, case) + ".pvd")
    
    # Next, we implement the main loop of the algorithm. In every iteration
    # we will halve the penalisation parameter and (re-)solve the
    # optimisation problem. The optimised control value will then be used as
    # an initial guess for the next optimisation problem.
    #
    # We begin by defining the loop and updating the :math:`\alpha` value.


    for i in range(4):
        # Update the penalisation value
        rho.assign(float(rho)/2)
        print("Set alpha to %f." % float(rho))
        
        # We rely on a useful property of dolfin-adjoint here: if an object
        # has been used while being a Placeholder (here achieved by creating the
        # :py:class:`Placeholder <pyadjoint.placeholder.Placeholder>` object
        # above), dolfin-adjoint does not copy that object, but
        # keeps a reference to it instead.  That means that assigning a new
        # value to ``rho`` has the effect that the optimisation routine will
        # automatically use that new value.
        #
        # Next we solve the optimisation problem for the current ``rho``.  We
        # use the ``L-BFGS-B`` optimisation algorithm here :cite:`zhu1997b` and
        # select a set of sensible stopping criteria:
        
        alpha_opt = minimize(Jhat, method="L-BFGS-B", bounds=(lb_alpha, ub_alpha), options={"gtol": gtol, "ftol": 1e-100, "disp": True, "maxfun":1500, "maxiter":1500, "maxls":30, "eps":1e-8, "iprint":101})
        #NOTE: The options are for the optimization process
        # "maxiter" : Maximum number of iterations (optimization process).
        #    iprint : int, optional
        #    Controls the frequency of output. iprint < 0 means no output; iprint = 0 print only one line at the last iteration;
        #0 < iprint < 99 print also f and |proj g| every iprint iterations;
        #iprint = 99 print details of every iteration except n-vectors;
        #iprint = 100 print also the changes of active set and final x;
        #iprint > 100 print details of every iteration including x and g.

        # The following step is optional and implements a performance
        # improvement. The idea is to use the optimised state solution as an
        # initial guess for the Newton solver in the next optimisation round.
        # It demonstrates how one can access and modify variables on the
        # ``dolfin-adjoint`` tape.
        #
        # First, we extract the optimised state (the ``u`` function) from the
        # tape. This is done with the ``Control.tape_value()``
        # function. By default it returns the last known iteration of that
        # function on the tape, which is exactly what we want here:

        u_opt = Control(u).tape_value()
    
        # The next line modifies the tape such that the initial guess for ``u``
        # (to be used in the Newton solver in the forward problem) is set to
        # ``u_opt``.  This is achieved with the
        # :py:func:`Control.update
        # <dolfin_adjoint.Control.update>` function and the initial guess control defined earlier:

        ic.update(u_opt)


        # Finally, we store the optimal state and control to disk and print some
        # statistics:

        u_pvd << u_opt
        alpha_pvd << alpha_opt
        # Save the optimal solutions
        save_opt_sol(alpha_opt, u_opt, para, savedir, mesh, i)
        
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
#        alpha_opt_norm2 = sqrt(assemble(inner(alpha_opt, alpha_opt)*dx))
        alpha_opt_norm2 = norm(alpha_opt)
        
        
        # summary results to a text file
        write_results = open("chaos2_summary_DA_results_2021_ex%s%s.txt"%(example, case), "a+")
        write_results.write("\n Summary results of example {:d}, case {:d} with rho_{:d}" .format(example, case, i))
        write_results.write("\n Objective value = {:.10f}" .format (work + compliance + perimeter_penalization) )
        write_results.write("\n Work_plus_compliance = {:.10f}" .format(work+compliance) )
        write_results.write("\n Work = {:.10f} \r\n" .format(work) )
        write_results.write("\n Compliance = {:.10f}" .format(compliance) )
        write_results.write("\n Perimeter_penalization = {:.10f}" .format(perimeter_penalization) )
        write_results.write("\n Vol_alpha = {:.10f}" .format(vol_alpha_opt) )
        write_results.write("\n Vol_ratio_percentage = {:.10f}" .format(vol_ratio_percentage) )
        write_results.write("\n Alpha_norm2 = {:.10f}"  .format(alpha_opt_norm2) )
        write_results.close()

        print("\nwork = {:.8f}; compliance = {:.8f}" .format(work, compliance))
        print("\ndissipated_energy = {:.8f}" .format(perimeter_penalization))
        print("Norm of u_opt: %s" % norm(u_opt))
#        print("Norm of u_opt: %s" % sqrt(assemble(inner(u_opt, u_opt)*dx)))
        print("Norm of alpha_opt: %s" % alpha_opt_norm2)
        print("Max of alpha_opt: %s " % alpha_opt_max)
        print("Min of alpha_opt: %s " % alpha_opt_min)
        print("min/max u_opt_x:", u1.vector().min(), u1.vector().max())
        print("min/max u_opt_y:", u2.vector().min(), u2.vector().max())

        # Save the optimal stress
        save_opt_stress = para["post_processing"]["save_opt_stress"];
        if save_opt_stress:
            def sigma(u):
                return phi(alpha, p, delta)* sigma_0(u, mu, lambda_, d)
            
            V_sig = TensorFunctionSpace(mesh, "DG", degree=0)
            stress = project(sigma(u_opt), V_sig)
            File(savedir + "/opt_sol%s/stress%s_rho%s.pvd"%(case, case, i)) << stress
        
    return

# Define the smooth approximation :math:`\max_{\epsilon}` of
# the maximum operator:
def smoothmax(r, eps=1e-4):
    return conditional(gt(r, eps), r - eps/2, conditional(lt(r, 0), 0, r**2 / (2*eps)))


# Define strain and stress
def epsilon(u):
    return sym(nabla_grad(u))

def sigma_0(u, mu, lambda_, d):
    return lambda_*tr(epsilon(u))*Identity(d) + 2*mu*epsilon(u)

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


def boundary_condition(V_pde, sub_domains, example):
    [sub_left_1, sub_left_2, sub_top_3, sub_top_4, sub_right_5, sub_right_6, sub_right_7, sub_bottom_8, sub_bottom_9] = sub_domains
    
    # Dirichlet B.C. for displacement field
    bcl_1_pde = DirichletBC(V_pde, Constant((0., 0.)), sub_left_1);
    bcl_2_pde = DirichletBC(V_pde, Constant((0., 0.)), sub_left_2);
    bcr_5_pde = DirichletBC(V_pde.sub(0), Constant(0.), sub_right_5);
    bcr_6_pde = DirichletBC(V_pde.sub(0), Constant(0.), sub_right_6);
    bcr_7_pde = DirichletBC(V_pde.sub(0), Constant(0.), sub_right_7);
    
    bcs_pde = [bcl_1_pde, bcl_2_pde, bcr_5_pde, bcr_6_pde, bcr_7_pde]
    # Note: The displacement is fixed in the x-direction on the right boundary and clamped on the left boundary.
    return bcs_pde


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
        # traction force is applied on the top boundary
        g_L1 = Constant((0., 0.)); g_L2 = Constant((0., 0.));
        g_T3 = Constant((0., 0.)); g_T4 = Constant((0., 0.));
        g_R5 = Constant((0., 0.)); g_R6 = Constant((gx, gy)); g_R7 = Constant((0., 0.));
        g_B8 = Constant((0., 0.)); g_B9 = Constant((0., 0.));
        print("Traction forces: g_L1 = (0, 0); g_L2 = (0, 0); g_T3 = (0, 0); g_T4 = (0, 0); g_R5 = (0, 0); g_R6 = ({:.3f}, {:.3f}); g_R7 = (0, 0); g_B8 = (0, 0);  g_B9 = (0, 0)" .format(gx, gy))
    else:
        # traction force is applied on the right-center boundary
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
    # Set constraints for alpha
    #-----------------------------------------------------------------
    if example == 3:
        # Two obstacles case. Define my functions for constraints
        #------------------------------------------
        class my_ub(UserExpression):
            def __init__(self, ub_u_y, x1, x2, Lx, Ly, **args):
                super().__init__(self,**args)
                self.ub_u_y = ub_u_y; self.x1 = x1; self.x2 = x2
                self.Lx = Lx; self.Ly = Ly;
            def eval_cell(self, values, x, cell):
                if near(x[1], self.Ly, DOLFIN_EPS) and (self.x1 < x[0]) and (x[0] < self.x2):
                    values[0] = 10.; values[1] = self.ub_u_y - x[1];
                else:
                    values[0] = 10.; values[1] = 10.; # infty := 10
            def value_shape(self):
                return (2,)
        
        class my_lb(UserExpression):
            def __init__(self, lb_u_y, x1, x2, Lx, Ly, **args):
                super().__init__(self,**args)
                self.lb_u_y = lb_u_y; self.x1 = x1; self.x2 = x2
                self.Lx = Lx; self.Ly = Ly;
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
                super().__init__(self,**args)
                self.lb_alpha = lb_alpha; self.ub_alpha = ub_alpha;
                self.Lx = Lx; self.Ly = Ly;
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


def save_opt_sol(alpha_opt, u_opt, para, savedir, mesh, i):
    option = "opt_pp" # opt_post_processing_parameters
    case = para["problem"]["case"];
    [save_opt_u, save_opt_active_set_ub, save_opt_active_set_lb, save_opt_u_adj, save_opt_alpha, save_opt_stress] = get_post_processing_parameters(para, option)
    
    if save_opt_u:
        File(savedir + "/opt_sol%s/u_opt%s_rho%s.pvd"%(case, case, i))<< u_opt;
        xdmf_u_opt = XDMFFile(mesh.mpi_comm(), savedir + "/opt_sol%s/u_opt%s_rho%s.xdmf"%(case, case, i))
        with xdmf_u_opt as file_u_opt:
            file_u_opt.write_checkpoint(u_opt, "u_opt")
    
    if save_opt_alpha:
        File(savedir + "/opt_sol%s/alpha_opt%s_rho%s.pvd"%(case, case, i))<< alpha_opt;
        xdmf_alpha_opt = XDMFFile(mesh.mpi_comm(), savedir + "/opt_sol%s/alpha_opt%s_rho%s.xdmf"%(case, case, i))
        with xdmf_alpha_opt as file_alpha_opt:
            file_alpha_opt.write_checkpoint(alpha_opt, "alpha_opt")
    return


if __name__== "__main__":
    start = time. time()
    main()
    end = time. time()
    print("\nExecution time (in seconds):",(end - start))
