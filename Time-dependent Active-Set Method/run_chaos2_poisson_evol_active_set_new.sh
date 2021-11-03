#!/bin/bash
##################################################################################
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 101 --time_app.nt 10 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 100. --GDA.iter_max 3000 --GDA.search_scheme non_constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 5. |& tee -a output_evol_poisson_active_set_case101.txt

#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 100 --time_app.nt 10 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 100. --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case100.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 101 --time_app.nt 20 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 100. --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case101.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 102 --time_app.nt 30 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 100. --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case102.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 103 --time_app.nt 40 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 100. --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case103.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 104 --time_app.nt 50 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 100. --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case104.txt

##on chaos 3, different initial guess
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 110 --time_app.nt 20 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 75 --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case110.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 111 --time_app.nt 20 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 50 --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case111.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 112 --time_app.nt 20 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 150 --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case112.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 113 --time_app.nt 20 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 200 --GDA.iter_max 5000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case113.txt

##example 2, control function just move from left to right
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 2 --problem.case 200 --time_app.nt 20 --const.gamma 1. --const.rho 1e-3 --initial_guess.is_constant 1 --initial_guess.slope 25. --GDA.iter_max 5000 --GDA.tol 1e-4 --GDA.search_scheme constant_step_size --GDA.save_frequency 100 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case200.txt

#mpirun --np 8 python3 evol_poisson_active_set.py --ex 2 --problem.case 201 --time_app.nt 20 --const.gamma 1. --const.rho 1e-2 --initial_guess.is_constant 1 --initial_guess.slope 25. --GDA.iter_max 5000 --GDA.tol 1e-4 --GDA.search_scheme constant_step_size --GDA.save_frequency 100 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case201.txt

#example 3, control function just move from left to right
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 3 --problem.case 300 --time_app.nt 20 --const.gamma 1. --const.rho 1e-3 --initial_guess.is_constant 1 --initial_guess.slope 5. --GDA.iter_max 5000 --GDA.tol 1e-4 --GDA.search_scheme constant_step_size --GDA.save_frequency 100 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case300.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 3 --problem.case 301 --time_app.nt 20 --const.gamma 1. --const.rho 1e-2 --initial_guess.is_constant 1 --initial_guess.slope 5. --GDA.iter_max 5000 --GDA.tol 1e-4 --GDA.search_scheme constant_step_size --GDA.save_frequency 100 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case301.txt


# Rerun the problem with more iterations
# example 1
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 1000 --time_app.nt 20 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 100. --GDA.iter_max 10000 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case1000.txt

# example 2: three time step
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 2 --problem.case 2000 --time_app.nt 3 --const.gamma 1. --const.rho 1e-3 --initial_guess.is_constant 1 --initial_guess.slope 50. --GDA.iter_max 10000 --GDA.tol 1e-4 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case2000.txt
#
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 2 --problem.case 2001 --time_app.nt 3 --const.gamma 1. --const.rho 1e-2 --initial_guess.is_constant 1 --initial_guess.slope 50. --GDA.iter_max 10000 --GDA.tol 1e-4 --GDA.search_scheme constant_step_size --GDA.save_frequency 500 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case2001.txt

# Rerun the problem with more iterations, FEM of quadratic appeoxmation
# example 1
mpirun --np 8 python3 evol_poisson_active_set.py --ex 1 --problem.case 10 --time_app.nt 20 --const.gamma 1. --const.rho 1e-15 --initial_guess.slope 100. --GDA.iter_max 10000 --GDA.search_scheme constant_step_size --GDA.save_frequency 1000 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case10.txt

# example 2: three time step
#mpirun --np 8 python3 evol_poisson_active_set.py --ex 2 --problem.case 20 --time_app.nt 3 --const.gamma 1. --const.rho 1e-15 --initial_guess.is_constant 1 --initial_guess.slope 50. --GDA.iter_max 10000 --GDA.tol 1e-4 --GDA.search_scheme constant_step_size --GDA.save_frequency 1000 --GDA.step_size 1. --GDA.step_size_min 1e-3 --GDA.step_size_max 10. |& tee -a output_evol_poisson_active_set_case20.txt
