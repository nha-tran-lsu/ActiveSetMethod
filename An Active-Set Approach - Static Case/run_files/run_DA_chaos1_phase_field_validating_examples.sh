#!/bin/bash
####################################################################
### delta=1e-3
####################################################################
###################################################################
## Validating examples (active-set method vs penalty method)
###################################################################
#Ex3
#mpiexec -n 8 python3 DA_phase_field_opt.py --ex 3 --problem.case 7200 --gtol 5.e-8 --problem.initial_guess.is_constant 0 --problem.initial_guess.sin_px 2 --problem.initial_guess.sin_py 2 --problem.const.delta 1.e-3 --mesh.structured_mesh 1 --mesh.cell_size 0.005 --material.ell 0.025  --problem.obstacle.lb_u_y -0.035 --problem.obstacle.ub_u_y 0.25 --problem.const.eta 0.3 --problem.const.kappa 5.e-5 --problem.force.gy -10 post_processing.TAO_saving_frequency 200|& tee -a chao1_DA_ex37200_validating_delta_1e3.txt

mpiexec -n 8 python3 DA_phase_field_opt.py --ex 3 --problem.case 7201 --gtol 5.e-8 --problem.initial_guess.is_constant 0 --problem.initial_guess.sin_px 2 --problem.initial_guess.sin_py 2 --problem.const.delta 1.e-3 --mesh.structured_mesh 1 --mesh.cell_size 0.005 --material.ell 0.025  --problem.obstacle.lb_u_y -0.035 --problem.obstacle.ub_u_y 0.25 --problem.const.eta 0.35 --problem.const.kappa 5.e-5 --problem.force.gy -10 post_processing.TAO_saving_frequency 200|& tee -a chao1_DA_ex37201_validating_delta_1e3.txt

#Ex2
#mpiexec -n 8 python3 DA_phase_field_opt.py --ex 2 --problem.case 7100 --gtol 5.e-8 --problem.initial_guess.is_constant 0 --problem.initial_guess.sin_px 10 --problem.initial_guess.sin_py 10 --problem.const.delta 1.e-3 --mesh.structured_mesh 1 --mesh.cell_size 0.005 --material.ell 0.025  --problem.obstacle.lb_u_y -0.02 --problem.obstacle.ub_u_y 1e8 --problem.const.eta 0.08 --problem.const.kappa 5.e-5 --problem.force.gy -1 post_processing.TAO_saving_frequency 200|& tee -a chao1_DA_ex27100_validating_delta_1e3.txt
#Ex2.A
mpiexec -n 8 python3 DA_phase_field_opt.py --ex 2 --problem.case 7110 --gtol 5.e-8 --problem.initial_guess.is_constant 0 --problem.initial_guess.sin_px 10 --problem.initial_guess.sin_py 10 --problem.const.delta 1.e-3 --mesh.structured_mesh 1 --mesh.cell_size 0.005 --material.ell 0.025  --problem.obstacle.lb_u_y -1e8 --problem.obstacle.ub_u_y 1e8 --problem.const.eta 0.3 --problem.const.kappa 5.e-5 --problem.force.gy -1 post_processing.TAO_saving_frequency 200|& tee -a chao1_DA_ex27110_validating_delta_1e3.txt
##Ex1
#mpiexec -n 8 python3 DA_phase_field_opt.py --ex 1 --problem.case 7000 --gtol 5.e-8 --problem.initial_guess.is_constant 1 --problem.initial_guess.sin_px 10 --problem.initial_guess.sin_py 10 --problem.const.delta 1.e-3 --mesh.structured_mesh 1 --mesh.cell_size 0.005 --material.ell 0.025  --problem.obstacle.lb_u_y -1e8 --problem.obstacle.ub_u_y 1e8 --problem.const.eta 0.3 --problem.const.kappa 5.e-5 --problem.force.gy -1 post_processing.TAO_saving_frequency 1000|& tee -a chao1_DA_ex17000_validating_delta_1e3.txt
