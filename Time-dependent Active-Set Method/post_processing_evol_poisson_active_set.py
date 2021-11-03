from dolfin import *
from collections import OrderedDict
import numpy as np
import time, logging, os
import errno
from petsc4py import PETSc
import matplotlib.pyplot as plot
import sympy as sym
import csv
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
    # Plot the results of the first example
    save_dir = "ex1_figures/"
    path = "example_1_case_1000/fun_val_his_case_1000.txt"
    fig_name = "ex1_relative_objective_semilog.pdf"
    semilog_plot_f(path, save_dir, fig_name)

    fig_name = "ex1_relative_objective_loglog.pdf"
    loglog_plot_f(path, save_dir, fig_name)

    fig_name = "ex1_relative_gradient_loglog.pdf"
    loglog_plot_g(path, save_dir, fig_name)

    fig_name = "ex1_relative_gradient_semilog.pdf"
    semilog_plot_g(path, save_dir, fig_name)

    path = "example_1_case_1000/"
    fig_name = "ex1_dis_over_time.pdf"
    file_data = "example_1_case_1000/fun_val_his_case_1000_max_pre_dis_overtime.csv"
    file_opt = "example_1_case_1000/fun_val_his_case_1000_max_opt_dis_overtime.csv"
    plot_max_dis(path, file_data, file_opt, save_dir, fig_name)

    # Plot the results of the second example
    save_dir = "ex2_figures/"
    path = "example_2_case_2000/fun_val_his_case_2000.txt"
    fig_name = "ex2_relative_objective_semilog.pdf"
    semilog_plot_f(path, save_dir, fig_name)

    fig_name = "ex2_relative_objective_loglog.pdf"
    loglog_plot_f(path, save_dir, fig_name)

    fig_name = "ex2_relative_gradient_semilog.pdf"
    semilog_plot_g(path, save_dir, fig_name)

    fig_name = "ex2_relative_gradient_loglog.pdf"
    loglog_plot_g(path, save_dir, fig_name)
    
    
    return


def plot_max_dis(path, file_data, file_opt, save_dir, fig_name):
    # define the name of the directory to be created
    try:
        os.mkdir(save_dir)
    except OSError:
        print ("Creation of the directory %s failed" % save_dir)
    else:
        print ("Successfully created the directory %s" % save_dir)

    # Read prescribed displacement
    t_ = []
    d_max = []
    with open(file_data,'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        next(csvreader) # This skips the first row of the CSV file.
        
        for row in csvreader:
            t_.append(row[0])
            d_max.append(float(row[1]))

    # Read optimal displacement
    t_ = []
    u_max = []
    with open(file_opt,'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        next(csvreader) # This skips the first row of the CSV file.
        
        for row in csvreader:
            t_.append(row[0])
            u_max.append(float(row[1]))

    # Analytic solution
    # u(t, x = 0.5) = 25*t,   for t <= t_0 = (1-sqrt(2)/2)*T = 0.2928932188
    # u(t, x = 0.5) = 25*t_0, for t > t_0 = (1-sqrt(2)/2)*T
    t_star = 0.2928932188
    def u(t):
        if(t <= t_star): return 25*t
        else: return 25*t_star
    
#    tt = np.arange(0., 1.05, 0.05)
    tt = np.linspace(0., 1., 21)
    u_exact = []
    for i in range(len(tt)):
        u_exact.append(u(tt[i]))
    
#    print("u_exact = ", u_exact)
#    print("tt = ", tt)
    plot.clf()
    fig = plot.figure()
    p1 =plot.plot(t_, u_exact, color='b', linestyle='dashed', linewidth=1, marker='*', markerfacecolor='r', markersize=3, label = "u_exact(t, x = 0.5)")
    p1 =plot.plot(t_, d_max, color='g', linestyle='dotted', linewidth=1, marker='.', markerfacecolor='g', markersize=3, label = "d(t, x = 0.5)")
    p1 =plot.plot(t_, u_max, color='k', linestyle='-', linewidth=1, marker='o', markerfacecolor='r', markersize=3, label = "u_opt(t, x = 0.5)")
    
    plot.xticks(rotation = 45)
    plot.xlabel('Time')
    plot.ylabel('Displacement')
    plot.legend()
    # Draw grid lines with red color and dashed style
    plot.grid(color='grey', linestyle='-.', linewidth=0.2)

#    plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    plot.subplots_adjust(left=0.1, bottom=0.15)

    # Save the plot
    fig.savefig(save_dir + "%s"%(fig_name))
    return
   
   

def semilog_plot_f(path, save_dir, fig_name):
    # define the name of the directory to be created
    try:
        os.mkdir(save_dir)
    except OSError:
        print ("Creation of the directory %s failed" % save_dir)
    else:
        print ("Successfully created the directory %s" % save_dir)

    iter = []
    f = []
    for line in open(path, 'r'):
        lines = [i for i in line.split()]
        iter.append(int(float(lines[0])))
        f.append(float(lines[1]))

    n = len(f)
    ff = []
    for x in f:
        ff.append(x/f[0])
    plot.clf()
    fig = plot.figure()

    # Display grid
    plot.grid(True, which="both")
    # Linear X axis, Logarithmic Y axis
    p1 = plot.semilogy(iter[:(n - 1)], ff[:(n - 1)])
    # Provide the title for the semilog plot
    #plot.title('Relative Gradient')

    # Give x axis label for the semilog plot
    plot.xlabel('Iterations')

    # Give y axis label for the semilog plot
    plot.ylabel('Cost')
#    plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    plot.subplots_adjust(left=0.15)

    # Save the plot
    fig.savefig(save_dir + "%s"%(fig_name))

    return
   
 
def loglog_plot_f(path, save_dir, fig_name):
    # define the name of the directory to be created
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % save_dir)
    else:
        print ("Successfully created the directory %s" % save_dir)

    iter = []
    f = []
    for line in open(path, 'r'):
        lines = [i for i in line.split()]
        iter.append(int(float(lines[0])))
        f.append(float(lines[1]))

    n = len(f)
    ff = []
    for x in f:
        ff.append(x/f[0])

    plot.clf()
    fig = plot.figure()

    # Display grid
    plot.grid(True, which="both")
    # Linear X axis, Logarithmic Y axis
    p1 = plot.loglog(iter[:(n - 1)], ff[:(n - 1)])

    # Provide the title for the semilog plot
    #plot.title('Relative Gradient')

    # Give x axis label for the semilog plot
    plot.xlabel('Iterations')

    # Give y axis label for the semilog plot
    plot.ylabel('Cost', fontsize=11)
    plot.subplots_adjust(left=0.15)
#    plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    # Save the plot

    fig.savefig(save_dir + "%s"%(fig_name))

    return
  

def semilog_plot_g(path, save_dir, fig_name):
    # define the name of the directory to be created
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % save_dir)
    else:
        print ("Successfully created the directory %s" % save_dir)

    iter = []
    f = []
    for line in open(path, 'r'):
        lines = [i for i in line.split()]
        iter.append(int(float(lines[0])))
        f.append(float(lines[2]))

    n = len(f)
    ff = []
    for x in f:
        ff.append(x/f[0])

    plot.clf()
    fig = plot.figure()

    # Display grid
    plot.grid(True, which="both")
    # Linear X axis, Logarithmic Y axis
    p1 = plot.semilogy(iter[:(n - 1)], ff[:(n - 1)])

    # Provide the title for the semilog plot
    #plot.title('Relative Gradient')

    # Give x axis label for the semilog plot
    plot.xlabel('Iterations')

    # Give y axis label for the semilog plot
    plot.ylabel('Relative Gradient')

    # Save the plot
    fig.savefig(save_dir + "%s"%(fig_name))
    return
    
def loglog_plot_g(path, save_dir, fig_name):
    # define the name of the directory to be created
    try:
        os.mkdir(save_dir)
    except OSError:
        print ("Creation of the directory %s failed" % save_dir)
    else:
        print ("Successfully created the directory %s" % save_dir)

    iter = []
    f = []
    for line in open(path, 'r'):
        lines = [i for i in line.split()]
        iter.append(int(float(lines[0])))
        f.append(float(lines[2]))

    n = len(f)
    ff = []
    for x in f:
        ff.append(x/f[0])

    plot.clf()
    fig = plot.figure()

    # Display grid
    plot.grid(True, which="both")
    #plot.grid(True, which="both", color='grey', linestyle='-.', linewidth=0.5)
    #plot.grid(True, which="both", linestyle='-.', linewidth=0.5)
    # Linear X axis, Logarithmic Y axis
    p1 = plot.loglog(iter[:(n - 1)], ff[:(n - 1)])
    #p1 = plot.loglog(iter[:(val_len - 1)], rel_val[:(val_len - 1)], color='k', linestyle='-', linewidth=1, marker='*', markerfacecolor='red', markersize=3)
    #plot.legend("gradient_value%s"%fig_name)

    # Provide the title for the semilog plot
    #plot.title('Relative Gradient Loglog Plot')

    # Give x axis label for the semilog plot
    plot.xlabel('Iterations')

    # Give y axis label for the semilog plot
    plot.ylabel('Relative Gradient')

    # Save the plot
    fig.savefig(save_dir + "%s"%(fig_name))
    return
if __name__== "__main__":
    start = time. time()
    main()
    end = time. time()
    print("\nExecution time (in seconds):",(end - start))


