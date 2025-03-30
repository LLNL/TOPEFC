import firedrake as fd
from firedrake import inner, grad, div, dot, jump, dx, ds, dS
from firedrake.petsc import PETSc
import firedrake_adjoint as fda
import ufl
from ufl.algebra import Abs as abs
from pyMMAopt import ReducedInequality, MMASolver
import itertools
import os
import signal
from mpi4py import MPI
import numpy as np
import argparse
from pyop2.mpi import COMM_WORLD
import pandas
from firedrake.tsfc_interface import TSFCKernel
from pyop2.global_kernel import GlobalKernel
import gc
import petsc4py
from scipy.integrate import simpson

parprint = PETSc.Sys.Print

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--input_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str, default="./plots")
parser.add_argument("--dimless_entry", type=int, default=1)

parser.add_argument("--mod_brugg", type=float, default=0.02)
parser.add_argument("--engy_cval", type=float, default=0.5)
parser.add_argument("--sc_weight", type=float, default=1.0)
parser.add_argument("--beta_scale", type=float, default=0.0)
parser.add_argument("--delta_cont", type=float, default=1.0)
parser.add_argument("--p_init", type=float, default=1.)

parser.add_argument("--Lx", type=float, default=1.)
parser.add_argument("--Nx", type=int, default=50)
parser.add_argument("--loop_init", type=int, default=300)
parser.add_argument("--opt_cycle", type=int, default=0)
parser.add_argument("--max_iter", type=int, default=300)
parser.add_argument("--opt_strat", type=int, default=0)

parser.add_argument("--solver", type=int, default=0)
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--forward", type=int, default=0)
parser.add_argument("--chk_init", type=int, default=0)
parser.add_argument("--movlim", type=float, default=0.01)

parser.add_argument("--q_ind", type=float, default=100.)
parser.add_argument("--rho_init", type=str, default='constant')
parser.add_argument("--rho_init_val", type=float, default=0.5)
parser.add_argument("--filter", type=int, default=1)
parser.add_argument("--proj", type=float, default=0.)

parser.add_argument("--force_break", type=int, default=0)
parser.add_argument("--break_offset", type=float, default=0.02)
parser.add_argument("--sim_cycle", type=str, default="half")
parser.add_argument("--cs_max", type=float, default=1.0)

parser.add_argument("--conc", type=int, default=1)
parser.add_argument("--csurf", type=int, default=0)
parser.add_argument("--run_cv", type=int, default=0)
parser.add_argument("--cv_file", type=str, default="cv_sim_data.csv")

args, _ = parser.parse_known_args()

q_ind = fd.Constant(args.q_ind)
p_cond = fd.Constant(1.5)
p_engy_opt = fd.Constant(3.)

if args.opt_strat == 0 or args.opt_strat == 1:
    p_cond_opt = fd.Constant(args.p_init)
else:
    p_cond_opt = fd.Constant(1.)

if args.opt_strat == 2 or args.opt_strat == 3:
    p_mod_brugg = fd.Constant(0.02)
else:
    p_mod_brugg = fd.Constant(args.mod_brugg)

sc_weight = fd.Constant(args.sc_weight)
engy_cval = args.engy_cval
beta_scale = fd.Constant(args.beta_scale)
offset = args.break_offset

def heaviside_projection(gamma, beta, eta):
    tanh = ufl.tanh
    return (tanh(beta*eta) + tanh(beta*(gamma-eta)))/(tanh(beta*eta) + tanh(beta*(1-eta)))

class SuperCapacitor:
    # Supercapacitor with redox reaction and binary symmetric charge electrolyte

    # linear solver parameters
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        #"snes_monitor": None,
        #"snes_converged_reason": None,
        #"ksp_monitor": None,
        #"ksp_converged_reason": None,
    }
    if args.solver == 1:
        solver_parameters["snes_rtol"] = 0
        solver_parameters["snes_atol"] = 1e-4
    if args.dim == 3:
        solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "gmres",
            "ksp_gmres_restart": 30,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            "fieldsplit_pc_type": "hypre",
            "fieldsplit_pc_hypre_boomeramg": {
            "strong_threshold": 0.7,
            "coarsen_type": "HMIS",
            "agg_nl": 3,
            "interp_type": "ext+i",
            "agg_num_paths": 5,
            },
            #"fieldsplit_pc_type": "lu",
            #"pc_factor_mat_solver_type": "mumps",
            # "snes_monitor": None,
            # "snes_converged_reason": None,
            # "ksp_monitor": None,
            # "ksp_converged_reason": None,
            "snes_type": "ksponly",
            "ksp_atol": 1e-4,
            "ksp_rtol": 0,
            "snes_rtol": 0,
            "snes_atol": 1e-4,
        }
        #solver_parameters = matfree_solver_parameters 
        if args.solver == 1:
            matfree_solver_parameters = {
                "mat_type": "matfree",
                "ksp_type": "gmres",
                "ksp_gmres_restart": 30,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative",
                "fieldsplit_pc_type": "python",
                "fieldsplit_pc_python_type": "firedrake.AssembledPC",
                "fieldsplit_assembled_pc_type": "hypre",
                "fieldsplit_assembled_pc_hypre_boomeramg": {
                "strong_threshold": 0.7,
                "coarsen_type": "HMIS",
                "agg_nl": 3,
                "interp_type": "ext+i",
                "agg_num_paths": 5,
                },
                # "snes_monitor": None,
                # "snes_converged_reason": None,
                # "ksp_monitor": None,
                # "ksp_converged_reason": None,
                "ksp_rtol": 1e-4,
                "snes_rtol": 0,
                "snes_atol": 1e-4,
            }
            solver_parameters = matfree_solver_parameters 
        elif args.solver == 2:
            solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "gmres",
                "ksp_gmres_restart": 30,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative",
                "fieldsplit_pc_type": "hypre",
                "fieldsplit_pc_hypre_boomeramg": {
                "strong_threshold": 0.7,
                "coarsen_type": "HMIS",
                "agg_nl": 3,
                "interp_type": "ext+i",
                "agg_num_paths": 5,
                },
                #"fieldsplit_pc_type": "lu",
                #"pc_factor_mat_solver_type": "mumps",
                # "snes_monitor": None,
                # "snes_converged_reason": None,
                # "ksp_monitor": None,
                # "ksp_converged_reason": None,
                "ksp_rtol": 1e-8,
                "snes_rtol": 0,
                "snes_atol": 1e-8,
            }



    iterative_param = {
        "mat_type": "aij",
        "ksp_type": "gmres",
        "ksp_gmres_restart": 30,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
        "fieldsplit_pc_type": "hypre",
        "ksp_rtol": 1e-4,
        "snes_rtol": 0,
        "snes_atol": 1e-4,
    }

    # optimization solver parameters
    mma_parameters = {
        "move": args.movlim,
        "maximum_iterations": args.max_iter,
        "m": 1,
        "IP": 0,
        "tol": 0.,
        "rfunctol": 0.,
        "accepted_tol": 1e-8,
        "gcmma": False,
        "norm": "L2",
        "output_dir": "./",
    }

    # constants
    F = 96485.33289         #Faraday's constant (C/mol)
    R = 8.3144598           #Ideal gas constant (J/mol/K)

    ####################################################################################
    #                                  Initialization                                  #
    ####################################################################################
    def __init__(self, input_file_name, dimless_entry = True):
        input_file = open(input_file_name, "r")
        self.entry_type = dimless_entry

        if (dimless_entry):
            for line in input_file:
                var_name, var_value = line.split("=")
                if var_name.strip() == "delta": self.delta = float(var_value.strip())
                if var_name.strip() == "beta": self.beta = float(var_value.strip())
                if var_name.strip() == "alpha": self.alpha = float(var_value.strip())
                if var_name.strip() == "lambda": self.lambd = float(var_value.strip())
                if var_name.strip() == "tplus": self.t_plus0 = float(var_value.strip())
                if var_name.strip() == "eta": self.eta = float(var_value.strip())
                if var_name.strip() == "tf": self.tf = float(var_value.strip())
                if var_name.strip() == "porosity": self.porosity = float(var_value.strip())
                if var_name.strip() == "total timestep":
                    self.tot_step = int(var_value.strip())
                if var_name.strip() == "filter radius":
                    self.rf = float(var_value.strip())
                if var_name.strip() == "filter scheme":
                    self.filter_scheme = var_value.strip()
                if var_name.strip() == "bd prop scheme":
                    self.bd_prop_scheme = var_value.strip()
                if var_name.strip() == "mesh type":
                    mesh_type = var_value.strip()

            if mesh_type == "quadrilateral":
                self.is_quad = True
            else:
                self.is_quad = False

            self.max_volt = self.tf * self.eta
            self.delta_redox = self.delta * self.beta
            self.delta_capac = self.delta * (1. - self.beta)
            self.t_minus0 = 1. - self.t_plus0
            self.dt_hat = self.tf / self.tot_step

            self.z_plus = 1.
            self.z_minus = -self.z_plus

            self.conc_conv = (self.z_plus * self.z_minus /
                    (self.t_plus0 * self.t_minus0 * (self.z_minus - self.z_plus)))

            # For t_hat = 1.0, delta = 2.0 is the most stable case
            # So continuation on delta is hardcoded to start from delta = 2.0
            if args.opt_strat == 1 or args.opt_strat == 3 or args.opt_strat == 5:
                self.delta_redox = fd.Constant(2.0 * self.beta)
                self.delta_capac = fd.Constant(2.0 * (1. - self.beta))
            else:
                self.delta_redox = fd.Constant(self.delta_redox)
                self.delta_capac = fd.Constant(self.delta_capac)

        else:
            F = SuperCapacitor.F
            R = SuperCapacitor.R
            for line in input_file:
                var_name, var_value = line.split("=")
                # scaling for nondimensionalization
                if var_name.strip() == "temperature": 
                    self.T = float(var_value.strip())
                if var_name.strip() == "characteristic length":
                    self.L = float(var_value.strip())
                if var_name.strip() == "final time":
                    self.Tfinal = float(var_value.strip())
                if var_name.strip() == "scan rate":
                    self.zeta = float(var_value.strip())
                if var_name.strip() == "specific capacitance":
                    self.C_d = float(var_value.strip())
                if var_name.strip() == "thermodynamic potential":
                    self.U_0 = float(var_value.strip())
                if var_name.strip() == "bulk concentration":
                    self.c_0 = float(var_value.strip())

                # additional material and reaction property inputs
                if var_name.strip() == "specific interfacial area":
                    self.a_0 = float(var_value.strip())
                if var_name.strip() == "bulk electric conductivity":
                    self.sigma_0 = float(var_value.strip())
                if var_name.strip() == "charge transfer coefficient":
                    self.alpha = float(var_value.strip())
                if var_name.strip() == "cation diffusion coefficient":
                    self.D_plus0 = float(var_value.strip())
                if var_name.strip() == "anion diffusion coefficient":
                    self.D_minus0 = float(var_value.strip())
                if var_name.strip() == "reaction rate constant":
                    self.k_rnx = float(var_value.strip())
                if var_name.strip() == "cation charge number":
                    self.z_plus = float(var_value.strip())
                if var_name.strip() == "anion charge number":
                    self.z_minus = float(var_value.strip())
                if var_name.strip() == "porosity":
                    self.porosity = float(var_value.strip())
                if var_name.strip() == "density":
                    self.edensity = float(var_value.strip())
                if var_name.strip() == "total timestep":
                    self.tot_step = int(var_value.strip())

                # optimization procedure parameters
                if var_name.strip() == "filter radius":
                    self.rf = float(var_value.strip())
                if var_name.strip() == "filter scheme":
                    self.filter_scheme = var_value.strip()
                if var_name.strip() == "boundary propagation scheme":
                    self.bd_prop_scheme = var_value.strip()
                if var_name.strip() == "mesh type":
                    mesh_type = var_value.strip()

            if mesh_type == "quadrilateral":
                self.is_quad = True
            else:
                self.is_quad = False

            # Compute dimensionless groups from physical inputs
            self.D_0 = (self.D_plus0 * self.D_minus0 * (self.z_plus - self.z_minus) /
                    (self.z_plus * self.D_plus0 - self.z_minus * self.D_minus0))
            self.u_plus0 = self.D_plus0 / (R * self.T)
            self.u_minus0 = self.D_minus0 / (R * self.T)
            self.tf = self.Tfinal * self.D_0 / (self.L * self.L)
            self.t_plus0 = (self.z_plus * self.u_plus0 /
                    (self.z_plus * self.u_plus0 - self.z_minus * self.u_minus0))
            self.t_plus0 = 0.5
            self.t_minus0 = 1. - self.t_plus0
            self.kappa_0 = (F * F * self.c_0 * self.z_plus * (-self.z_minus) *
                    (self.z_plus * self.u_plus0 - self.z_minus * self.u_minus0))
            self.i_0 = F * self.k_rnx * (self.c_0 ** (2 * self.alpha))
            self.dt_hat = self.tf / self.tot_step
            parprint("u_plus0 = ", self.u_plus0)
            parprint("u_minus0 = ", self.u_minus0)
            parprint("kappa_0 = ", self.kappa_0)
            parprint("i_0 = ", self.i_0)

            dredox = (self.a_0 * self.i_0 * (self.L ** 2) * F / (R * self.T) *
                    (1. / self.sigma_0 + 1. / self.kappa_0))
            dcapac = (self.a_0 * self.C_d * self.D_0 *
                    (1. / self.sigma_0 + 1. / self.kappa_0))
            self.delta = dredox + dcapac
            self.beta = dredox / self.delta

            # For physical inputs, user will provide delta_cont to setup the starting delta value
            # Default delta_cont value is set to 1.0
            if args.opt_strat == 1 or args.opt_strat == 3 or args.opt_strat == 5:
                self.delta_redox = fd.Constant(dredox * args.delta_cont)
                self.delta_capac = fd.Constant(dcapac * args.delta_cont)
            else:
                self.delta_redox = fd.Constant(dredox)
                self.delta_capac = fd.Constant(dcapac)

            self.lambd = self.kappa_0 / (self.sigma_0 + self.kappa_0)
            self.eta = self.zeta * F * self.L * self.L / (R * self.T * self.D_0)
            self.max_volt = self.tf * self.eta
            self.max_volt_dim = self.Tfinal * self.zeta

            self.conc_conv = (self.z_plus * self.z_minus /
                    (self.t_plus0 * self.t_minus0 * (self.z_minus - self.z_plus)))

        input_file.close()

        # Parameters to simulate charging phase
        self.cs_max = fd.Constant(args.cs_max)
        self.charge_dir = fd.Constant(-1.)

        # Simulate half cycle or full cycle
        if args.sim_cycle == "half":
            self.tot_cyc_step = self.tot_step
        elif args.sim_cycle == "full":
            self.tot_cyc_step = 2 * self.tot_step

        parprint("alpha = ", self.alpha)
        parprint("delta redox = ", self.delta_redox.dat.data[0])
        parprint("delta capac = ", self.delta_capac.dat.data[0])
        parprint("lambd = ", self.lambd)
        parprint("tplus = ", self.t_plus0)
        parprint("eta = ", self.eta)
        parprint("tf_hat = ", self.tf)
        parprint("dphi = ", self.eta * self.dt_hat)

    def setup_problem(self):
        # create mesh
        if args.dim == 2:
            self.mesh = fd.UnitSquareMesh(2*args.Nx, 2*args.Nx, quadrilateral = self.is_quad, name = "mesh_e")
        else:
            self.mesh = fd.CubeMesh(2*args.Nx, 2*args.Nx, 2*args.Nx, args.Lx, hexahedral = self.is_quad, name = "mesh_e")

        # setup function spaces
        D = fd.FunctionSpace(self.mesh, "DG", 0)
        V = fd.FunctionSpace(self.mesh, "CG", 1)

        # Whether to include fluid and solid concentration conservation equation in the system
        if args.csurf == 1 and args.conc == 1:
            W = V * V * V * V * D
        elif args.csurf == 0 and args.conc == 1:
            W = V * V * V * V
        else:
            W = V * V * V
        parprint(f"DOFs: {W.dim()}")

        # design variables
        self.rho = fd.Function(D, name = "rho")
        if args.dim == 2:
            self.x, self.y = fd.SpatialCoordinate(self.mesh)
        else:
            self.x, self.y, self.z = fd.SpatialCoordinate(self.mesh)

        if args.rho_init == "monolith":
            gap = 0.05
            self.rho.interpolate(ufl.conditional(ufl.Or(self.y <= 0.5-gap/2, self.y >= 0.5+gap/2),
                                                 fd.Constant(1.), fd.Constant(1e-8)))
            fd.File("rho.pvd").write(self.rho)
        elif args.rho_init.endswith(".h5"):
            with fd.HDF5File(args.rho_init, "r") as checkpoint:
                checkpoint.read(self.rho, "/control")
        else:
            self.rho.interpolate(fd.Constant(args.rho_init_val))

        # FEM variables
        self.u = fd.Function(W)
        self.u_n = fd.Function(W)
        self.v = fd.TestFunction(W)
        self.epsilon = fd.Function(D, name = "porosity")
        self.a_hat = fd.Function(D, name = "surface area")

        # system
        self.resid = None

        # solution variables
        self.phi_1c = None
        self.phi_1a = None
        self.phi_2 = None

        if args.conc == 1:
            self.conc = self.u[3]
        else:
            self.conc = fd.Constant(1.)

        if args.csurf == 1:
            self.csurf = self.u[4]
        else:
            self.csurf = fd.Constant(0.0)

    ####################################################################################
    #                                  Forward model                                   #
    ####################################################################################
    def i0_hat_cathode(self):
        return (self.cath_ind * self.sigma_hat + fd.Constant(1e-8)) * grad(self.u[0])

    def i0_hat_anode(self):
        return (self.ano_ind * self.sigma_hat + fd.Constant(1e-8)) * grad(self.u[1])

    def i1_hat(self):
        return self.conc * self.kappa_hat * grad(self.u[2])

    def dc_hat(self):
        return self.kappa_hat * grad(self.conc)

    def in_hat_cathode(self):
        return (
                (self.cath_ind * self.conc ** self.alpha) *
                (ufl.exp(self.alpha * (self.u[0] - self.u[2])) -
                 ufl.exp(-self.alpha * (self.u[0] - self.u[2])))
               )

    def in_hat_anode(self):
        return (
                (self.ano_ind * self.conc ** self.alpha) *
                (ufl.exp(self.alpha * (self.u[1] - self.u[2])) -
                 ufl.exp(-self.alpha * (self.u[1] - self.u[2])))
               )

    def ic_hat_cathode(self):
        return (
                (self.cath_ind * self.conc ** self.alpha *
                    (0.5 + self.charge_dir * (self.csurf / self.cs_max - 0.5))) *
                ((self.u[0] - self.u[2]) - (self.u_n[0] - self.u_n[2])) /
                fd.Constant(self.dt_hat)
                )

    def ic_hat_anode(self):
        return (
                (self.ano_ind  * self.conc ** self.alpha *
                    (0.5 + self.charge_dir * (self.csurf / self.cs_max - 0.5))) *
                ((self.u[1] - self.u[2]) - (self.u_n[1] - self.u_n[2])) /
                fd.Constant(self.dt_hat)
                )

    def electric_potential_cathode(self):
        self.resid = (
                inner(self.i0_hat_cathode(), grad(self.v[0])) * dx
                + fd.Constant(self.lambd) * self.delta_redox *
                  self.a_hat * self.in_hat_cathode() * self.v[0] * dx
                + fd.Constant(self.lambd) * self.delta_capac *
                  self.a_hat * self.ic_hat_cathode() * self.v[0] * dx
                )

    def electric_potential_anode(self):
        self.resid += (
                inner(self.i0_hat_anode(), grad(self.v[1])) * dx
                + fd.Constant(self.lambd) * self.delta_redox *
                  self.a_hat * self.in_hat_anode() * self.v[1] * dx
                + fd.Constant(self.lambd) * self.delta_capac *
                  self.a_hat * self.ic_hat_anode() * self.v[1] * dx
                )

    def ionic_potential(self):
        self.resid += (
                inner(self.i1_hat(), grad(self.v[2])) * dx
                - fd.Constant(1. - self.lambd) * self.delta_redox *
                  self.a_hat * self.in_hat_cathode() * self.v[2] * dx
                - fd.Constant(1. - self.lambd) * self.delta_redox *
                  self.a_hat * self.in_hat_anode() * self.v[2] * dx
                - fd.Constant(1. - self.lambd) * self.delta_capac *
                  self.a_hat * self.ic_hat_cathode() * self.v[2] * dx
                - fd.Constant(1. - self.lambd) * self.delta_capac *
                  self.a_hat * self.ic_hat_anode() * self.v[2] * dx
                )
        if args.conc == 1:
            self.resid += (fd.Constant(self.t_plus0 / self.z_plus + self.t_minus0 / self.z_minus) *
                    inner(self.dc_hat(), grad(self.v[2])) * dx)

    def species_conservation(self):
        self.resid += (
                fd.Constant(1. / self.dt_hat) * self.epsilon * self.conc * self.v[3] * dx
                - fd.Constant(1. / self.dt_hat) * self.epsilon * self.u_n[3] * self.v[3] * dx
                + inner(self.dc_hat(), grad(self.v[3])) * dx
                - fd.Constant(self.conc_conv * (1. - self.lambd) * self.t_minus0) * self.delta_redox *
                  self.a_hat * self.in_hat_cathode() * self.v[3] * dx
                - fd.Constant(self.conc_conv * (1. - self.lambd) * self.t_minus0) * self.delta_redox *
                  self.a_hat * self.in_hat_anode() * self.v[3] * dx
                - fd.Constant(self.conc_conv * (1. - self.lambd) * (-self.t_plus0)) * self.delta_capac *
                  self.a_hat * self.ic_hat_cathode() * self.v[3] * dx
                - fd.Constant(self.conc_conv * (1. - self.lambd) * self.t_minus0) * self.delta_capac *
                  self.a_hat * self.ic_hat_anode() * self.v[3] * dx
                )

    # Capability under development, currently only accounts for solid accumulation
    def surface_concentration(self):
        self.resid += (
                fd.Constant(1. / self.dt_hat) * self.u[4] * self.v[4] * dx
                - fd.Constant(1. / self.dt_hat) * self.u_n[4] * self.v[4] * dx
                + fd.Constant(self.conc_conv * (1. - self.lambd) * (-self.t_plus0)) * self.delta_capac *
                  self.a_hat * self.ic_hat_cathode() * self.v[4] * dx
                + fd.Constant(self.conc_conv * (1. - self.lambd) * self.t_minus0) * self.delta_capac *
                  self.a_hat * self.ic_hat_anode() * self.v[4] * dx
                )

    def forward(self, print_file=None):
        # Design initiation
        if args.rho_init == "monolith":
            self.rho_f = self.rho
        else:
            self.apply_filter()
        self.boundary_propagation()

        # interpolate cathode/anode indicator
        electrode_ind = (self.beta_prop + fd.Constant(1.)) / fd.Constant(2.)
        self.cath_ind = ((ufl.tanh(q_ind * fd.Constant(0.5)) - ufl.tanh(q_ind * (electrode_ind - fd.Constant(0.5)))) /
                (ufl.tanh(q_ind * fd.Constant(0.5)) - ufl.tanh(-q_ind * fd.Constant(0.5))))
        self.ano_ind = ((ufl.tanh(q_ind * (electrode_ind - fd.Constant(0.5))) - ufl.tanh(-q_ind * fd.Constant(0.5))) /
                (ufl.tanh(q_ind * fd.Constant(0.5)) - ufl.tanh(-q_ind * fd.Constant(0.5))))

        # interpolate porosity and conductivity
        sigma_hat1 = (1. - self.porosity) ** 1.5
        self.sigma_hat = fd.Constant(sigma_hat1) * self.rho_f ** p_cond
        self.sigma_opt = fd.Constant(sigma_hat1) * self.rho_f ** p_cond_opt

        kappa_hat1 = fd.Constant(self.porosity ** 1.5)
        self.kappa_hat = (kappa_hat1 * p_mod_brugg +
            (1. - kappa_hat1 * p_mod_brugg) * (1. - self.rho_f) ** p_cond)
        self.kappa_opt = (kappa_hat1 * p_mod_brugg +
            (1. - kappa_hat1 * p_mod_brugg) * (1. - self.rho_f) ** p_cond_opt)

        self.epsilon = fd.Constant(1.) - fd.Constant(1. - self.porosity) * self.rho_f
        self.a_hat = self.rho_f

        self.escale = abs(self.beta_prop) ** beta_scale

        # assemble_system
        self.electric_potential_cathode()
        self.electric_potential_anode()
        self.ionic_potential()
        if args.conc == 1:
            self.species_conservation()
        if args.csurf == 1:
            self.surface_concentration()

        # setup_solver
        D = fd.FunctionSpace(self.mesh, "DG", 1)
        W = self.u.function_space()
        voltage_bc1 = fd.Constant(0.)
        voltage_bc2 = fd.Constant(0.)
        bc1 = fd.DirichletBC(W.sub(1), voltage_bc1, 3)
        bc2 = fd.DirichletBC(W.sub(0), voltage_bc2, 4)
        boundary_condition = [bc1, bc2]
        problem = fd.NonlinearVariationalProblem(self.resid, self.u,
                bcs = boundary_condition)
        solver = fd.NonlinearVariationalSolver(problem,
           solver_parameters = SuperCapacitor.solver_parameters)
        self.solver = solver

        # PDE initial condition
        self.u.sub(0).assign(0.)
        self.u.sub(1).assign(0.)
        self.u.sub(2).assign(0.)
        if args.conc == 1:
            self.u.sub(3).assign(1.)
        if args.csurf == 1:
            self.u.sub(4).assign(0.)
        self.u_n.assign(self.u)

        # energy metric visualization variables
        self.eloss_dist = fd.Function(D, name = "E_loss")
        self.eredox_dist = fd.Function(D, name = "E_redox")
        self.ecapa_dist = fd.Function(D, name = "E_capac")

        t = 0.
        self.eloss = 0.
        self.rstored = 0.
        self.cstored = 0.
        self.engyin = 0.
        if print_file:
            if args.csurf == 1 and args.conc == 1:
                phi_1c, phi_1a, phi_2, conc, csurf = self.u.split()
                conc.rename("conc")
                csurf.rename("csurf")
            elif args.csurf == 0 and args.conc == 1:
                phi_1c, phi_1a, phi_2, conc = self.u.split()
                conc.rename("conc")
            else:
                phi_1c, phi_1a, phi_2 = self.u.split()

            phi_1c.rename("phi_1_cathode")
            phi_1a.rename("phi_1_anode")
            phi_2.rename("phi_2")
            if args.csurf == 1 and args.conc == 1:
                print_file.write(phi_1c, phi_1a, phi_2, conc, csurf, self.rho_f, self.beta_prop,
                        self.vel_prop, self.eloss_dist, self.eredox_dist, self.ecapa_dist, time=t)
            elif args.csurf == 0 and args.conc == 1:
                print_file.write(phi_1c, phi_1a, phi_2, conc, self.rho_f, self.beta_prop, self.vel_prop,
                        self.eloss_dist, self.eredox_dist, self.ecapa_dist, time=t)
            else:
                print_file.write(phi_1c, phi_1a, phi_2, self.rho_f, self.beta_prop, self.vel_prop,
                        self.eloss_dist, self.eredox_dist, self.ecapa_dist, time=t)

        for nstep in range(self.tot_cyc_step):
            t += self.dt_hat
            charge = True
            # charge / discharge
            if (nstep < self.tot_step):
                v_app = t * self.eta
            else:
                v_app = self.max_volt - (t - self.tf) * self.eta
                charge = False
            voltage_bc2.assign(v_app)

            solver.solve()

            # Compute energy stored and lost
            self.redox_stored(nstep, charge)
            self.capac_stored(nstep, charge)
            self.ohmic_loss(nstep)
            self.energy_input(nstep, v_app)

            self.eloss_dist += fd.interpolate(
                    (self.kappa_opt / fd.Constant(1. - self.lambd) *
                        self.conc * inner(grad(self.u[2]), grad(self.u[2])) +
                     self.sigma_opt / fd.Constant(self.lambd) * 
                        inner(grad(self.u[0]), grad(self.u[0])) +
                     self.sigma_opt / fd.Constant(self.lambd) *
                        inner(grad(self.u[1]), grad(self.u[1]))) * self.escale *
                    self.dt_hat, D)
            if args.conc == 1:
                self.eloss_dist += fd.interpolate(
                        self.kappa_opt / fd.Constant(1. - self.lambd) *
                        (self.t_plus0 / self.z_plus + self.t_minus0 / self.z_minus) *
                        inner(grad(self.conc), grad(self.u[2])) * self.escale * self.dt_hat, D)

            self.eredox_dist += fd.interpolate(self.escale *
                    self.delta_redox * (self.rho_f ** p_engy_opt) *
                    self.in_hat_cathode() * (self.u[0] - self.u[2]) * self.dt_hat, D)
            self.eredox_dist += fd.interpolate(self.escale *
                    self.delta_redox * (self.rho_f ** p_engy_opt) *
                    self.in_hat_anode() * (self.u[1] - self.u[2]) * self.dt_hat, D)
            self.ecapa_dist += fd.interpolate(self.escale *
                    self.delta_capac * (self.rho_f ** p_engy_opt) *
                    self.ic_hat_cathode() * (self.u[0] - self.u[2]) * self.dt_hat, D)
            self.ecapa_dist += fd.interpolate(self.escale *
                    self.delta_capac * (self.rho_f ** p_engy_opt) *
                    self.ic_hat_anode() * (self.u[1] - self.u[2]) * self.dt_hat, D)

            # Change parameters to discharging phase after last timestep of charging phase
            if nstep == self.tot_step - 1:
                self.charge_dir.assign(1.)
                energy_input = self.engyin
                redox_storage = self.rstored
                capac_storage = self.cstored
                if args.csurf == 0:
                    self.csurf.assign(args.cs_max)

            # March forward in time
            self.u_n.assign(self.u)
            parprint("Timestep ", nstep + 1, "Time", t)
            if print_file:
                if args.csurf == 1 and args.conc == 1:
                    phi_1c, phi_1a, phi_2, conc, csurf = self.u.split()
                    conc.rename("conc")
                    csurf.rename("csurf")
                elif args.csurf == 0 and args.conc == 1:
                    phi_1c, phi_1a, phi_2, conc = self.u.split()
                    conc.rename("conc")
                else:
                    phi_1c, phi_1a, phi_2 = self.u.split()
                phi_1c.rename("phi_1_cathode")
                phi_1a.rename("phi_1_anode")
                phi_2.rename("phi_2")
                if args.csurf == 1 and args.conc == 1:
                    print_file.write(phi_1c, phi_1a, phi_2, conc, csurf, self.rho_f, self.beta_prop,
                            self.vel_prop, self.eloss_dist, self.eredox_dist, self.ecapa_dist, time=t)
                elif args.csurf == 0 and args.conc == 1:
                    print_file.write(phi_1c, phi_1a, phi_2, conc, self.rho_f, self.beta_prop, self.vel_prop,
                            self.eloss_dist, self.eredox_dist, self.ecapa_dist, time=t)
                else:
                    print_file.write(phi_1c, phi_1a, phi_2, self.rho_f, self.beta_prop, self.vel_prop,
                            self.eloss_dist, self.eredox_dist, self.ecapa_dist, time=t)

        self.estored = self.rstored + self.cstored
        parprint("redox energy stored = ", redox_storage)
        parprint("capacitance energy stored = ", capac_storage)
        parprint("energy stored = ", redox_storage + capac_storage)

        energy_output = self.engyin - energy_input
        parprint("ohmic loss = ", self.eloss)
        parprint("energy input = ", energy_input)
        parprint("energy output = ", energy_output)
        parprint("percentage loss = ", self.eloss / energy_input)

        self.intermediate = fd.assemble(self.rho_f * (fd.Constant(1.) - self.rho_f) * dx)
        parprint("intermediate material = ", self.intermediate)

        parprint("short circuit indicator = ", self.short_circ)
        self.ebalance = fda.AdjFloat(2.) / (self.estored + self.engyin - self.eloss) + self.short_circ
        self.efficiency = fda.AdjFloat(1.) * self.eloss / self.engyin

        # Only use when setting forward = 1
        if (args.run_cv == 1 and args.sim_cycle == "full"):
            F = SuperCapacitor.F
            R = SuperCapacitor.R
            i_in_cath_dim = ((self.sigma_0 * R * self.T / (self.L * F)) *
                    dot(self.i0_hat_cathode(), fd.FacetNormal(self.mesh)))

            # sweep_to_low_potential
            t = 0.0;
            for nstep in range(self.tot_step):
                t += self.dt_hat
                v_app = -1.0 * t * self.eta
                voltage_bc2.assign(v_app)
                solver.solve()

                self.u_n.assign(self.u)

                i_app = fd.assemble(self.rho_f * i_in_cath_dim * self.L * self.L * ds(4))
                parprint("v_app = ", R * self.T / F * v_app, "i_app = ", i_app)

            parprint("Reached lowest potential")

            # start cv run
            self.currents = []
            self.potentials = []

            # record last current and potential
            self.currents.append(fd.assemble(self.rho_f * i_in_cath_dim *
                self.L * self.L * ds(4)))
            self.potentials.append(R * self.T / F * v_app)

            t = 0.0;
            tot_cv_step = self.tot_cyc_step * 2
            half_cv_step = self.tot_step * 2
            t_cv_final = self.tf * 2
            self.charge_dir.assign(-1.)
            if args.csurf == 0:
                self.csurf.assign(0.0)

            for nstep in range (tot_cv_step):
                t += self.dt_hat
                # charge / discharge
                if (nstep < half_cv_step):
                    v_app = -self.max_volt + t * self.eta
                else:
                    v_app = self.max_volt - (t - t_cv_final) * self.eta
                voltage_bc2.assign(v_app)
                solver.solve()

                self.u_n.assign(self.u)

                i_app = fd.assemble(self.rho_f * i_in_cath_dim * self.L * self.L * ds(4))
                parprint("v_app = ", R * self.T / F * v_app, "i_app = ", i_app)
                self.currents.append(i_app)
                self.potentials.append(R * self.T / F * v_app)

                if nstep == half_cv_step - 1:
                    parprint("Reached highest potential")
                    self.charge_dir.assign(1.)
                    if args.csurf == 0:
                        self.csurf.assign(args.cs_max)

            # Compute specific capacitance normalized by cathode mass
            cath_vol = fd.assemble(self.L ** 3 * self.rho_f * self.cath_ind * dx) * 1e+6
            cath_mass = cath_vol * self.edensity
            parprint("cathode mass (g) = ", cath_mass)
            cap_opt = (simpson(self.currents, x=self.potentials) /
                    (self.zeta * 2 * self.max_volt_dim * cath_mass))
            parprint("specific capacitance (F/g) = ", cap_opt)

            df = pandas.DataFrame({"potential" : self.potentials, "current" : self.currents})
            df.to_csv(args.cv_file, index=False)

    ####################################################################################
    #                                   Optimization                                   #
    ####################################################################################
    def energy_input(self, nstep, v_app):
        i_in_cath = dot(self.i0_hat_cathode(), fd.FacetNormal(self.mesh))
        i_in_ano = dot(self.i0_hat_anode(), fd.FacetNormal(self.mesh))
        if nstep == self.tot_step - 1 or nstep == self.tot_cyc_step - 1:
            self.engyin += (
                (fd.assemble(self.rho_f * ufl.sqrt(inner(i_in_cath, i_in_cath)) * ds(4))
                 + fd.assemble(self.rho_f * ufl.sqrt(inner(i_in_ano, i_in_ano)) * ds(3))) /
                (2 * self.lambd) * v_app * self.dt_hat * 0.5)
        else:
            self.engyin += (
                (fd.assemble(self.rho_f * ufl.sqrt(inner(i_in_cath, i_in_cath)) * ds(4))
                 + fd.assemble(self.rho_f * ufl.sqrt(inner(i_in_ano, i_in_ano)) * ds(3))) /
                (2 * self.lambd) * v_app * self.dt_hat)

    def ohmic_loss(self, nstep):
        # ignore initial condition since both potential is uniformly zero
        if nstep == self.tot_cyc_step - 1:
            self.eloss += ((
                fd.assemble(self.kappa_opt / fd.Constant(1. - self.lambd) * self.conc *
                    inner(grad(self.u[2]), grad(self.u[2])) * self.escale * dx) +
                fd.assemble(self.sigma_opt / fd.Constant(self.lambd) *
                    inner(grad(self.u[0]), grad(self.u[0])) * self.escale * dx) +
                fd.assemble(self.sigma_opt / fd.Constant(self.lambd) *
                    inner(grad(self.u[1]), grad(self.u[1])) * self.escale * dx)
                ) * self.dt_hat * 0.5)
            if args.conc == 1:
                self.eloss += (fd.assemble(
                        fd.Constant(self.t_plus0 / self.z_plus + self.t_minus0 / self.z_minus) *
                        self.kappa_opt / fd.Constant(1. - self.lambd) * inner(grad(self.conc), grad(self.u[2])) * 
                        self.escale * dx) * self.dt_hat * 0.5)
        else:
            self.eloss += ((
                fd.assemble(self.kappa_opt / fd.Constant(1. - self.lambd) * self.conc *
                    inner(grad(self.u[2]), grad(self.u[2])) * self.escale * dx) +
                fd.assemble(self.sigma_opt / fd.Constant(self.lambd) *
                    inner(grad(self.u[0]), grad(self.u[0])) * self.escale * dx) +
                fd.assemble(self.sigma_opt / fd.Constant(self.lambd) *
                    inner(grad(self.u[1]), grad(self.u[1])) * self.escale * dx)
                ) * self.dt_hat)
            if args.conc == 1:
                self.eloss += (fd.assemble(
                        fd.Constant(self.t_plus0 / self.z_plus + self.t_minus0 / self.z_minus) *
                        self.kappa_opt / fd.Constant(1. - self.lambd) * inner(grad(self.conc), grad(self.u[2])) * 
                        self.escale * dx) * self.dt_hat)

    def redox_stored(self, nstep, charge):
        if self.entry_type:
            if charge:
                # ignore initial condition since both potential is uniformly zero
                if nstep == self.tot_step - 1:
                    self.rstored += (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_cathode() * (self.u[0] - self.u[2]) * self.escale * dx) * self.dt_hat * 0.5)
                    self.rstored += (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_anode() * (self.u[1] - self.u[2]) * self.escale * dx) * self.dt_hat * 0.5)
                else:
                    self.rstored += (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_cathode() * (self.u[0] - self.u[2]) * self.escale * dx) * self.dt_hat)
                    self.rstored += (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_anode() * (self.u[1] - self.u[2]) * self.escale * dx) * self.dt_hat)
            else:
                if nstep == self.tot_cyc_step - 1:
                    self.rstored -= (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_cathode() * (self.u[0] - self.u[2]) * self.escale * dx) * self.dt_hat * 0.5)
                    self.rstored -= (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_anode() * (self.u[1] - self.u[2]) * self.escale * dx) * self.dt_hat * 0.5)
                else:
                    self.rstored -= (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_cathode() * (self.u[0] - self.u[2]) * self.escale * dx) * self.dt_hat)
                    self.rstored -= (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_anode() * (self.u[1] - self.u[2]) * self.escale * dx) * self.dt_hat)
        else:
            if charge:
                if nstep == self.tot_step - 1:
                    self.rstored += (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_cathode() * self.U_0 * self.escale * dx) * self.dt_hat * 0.5)
                    self.rstored -= (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_anode() * self.U_0 * self.escale * dx) * self.dt_hat * 0.5)
                else:
                    self.rstored += (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_cathode() * self.U_0 * self.escale * dx) * self.dt_hat)
                    self.rstored -= (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_anode() * self.U_0 * self.escale * dx) * self.dt_hat)
            else:
                if nstep == self.tot_cyc_step - 1:
                    self.rstored -= (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_cathode() * self.U_0 * self.escale * dx) * self.dt_hat * 0.5)
                    self.rstored += (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_anode() * self.U_0 * self.escale * dx) * self.dt_hat * 0.5)
                else:
                    self.rstored -= (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_cathode() * self.U_0 * self.escale * dx) * self.dt_hat)
                    self.rstored += (fd.assemble(self.delta_redox * (self.rho_f ** p_engy_opt) *
                        self.in_hat_anode() * self.U_0 * self.escale * dx) * self.dt_hat)

    def capac_stored(self, nstep, charge):
        if charge:
            self.cstored += (fd.assemble(self.delta_capac * fd.Constant(0.5) *
                (self.rho_f ** p_engy_opt) * (self.cath_ind * self.conc ** self.alpha) *
                    ((self.u[0] - self.u[2]) * (self.u[0] - self.u[2]) -
                        (self.u_n[0] - self.u_n[2]) * (self.u_n[0] - self.u_n[2])) * self.escale * dx))
            self.cstored += (fd.assemble(self.delta_capac * fd.Constant(0.5) *
                (self.rho_f ** p_engy_opt) * (self.ano_ind * self.conc ** self.alpha) *
                    ((self.u[1] - self.u[2]) * (self.u[1] - self.u[2]) -
                        (self.u_n[1] - self.u_n[2]) * (self.u_n[1] - self.u_n[2])) * self.escale * dx))
        else:
            self.cstored -= (fd.assemble(self.delta_capac * fd.Constant(0.5) *
                (self.rho_f ** p_engy_opt) * (self.cath_ind * self.conc ** self.alpha) *
                    ((self.u[0] - self.u[2]) * (self.u[0] - self.u[2]) -
                        (self.u_n[0] - self.u_n[2]) * (self.u_n[0] - self.u_n[2])) * self.escale * dx))
            self.cstored -= (fd.assemble(self.delta_capac * fd.Constant(0.5) *
                (self.rho_f ** p_engy_opt) * (self.ano_ind * self.conc ** self.alpha) *
                    ((self.u[1] - self.u[2]) * (self.u[1] - self.u[2]) -
                        (self.u_n[1] - self.u_n[2]) * (self.u_n[1] - self.u_n[2])) * self.escale * dx))

    def apply_filter(self):
        D = self.rho.function_space()
        if args.filter != 1:
            self.rho_f = fd.Function(D, name = "rho_f")
            self.rho_f.assign(self.rho)
            return
            

        if (self.filter_scheme == "CG"):
            # filter problem with CG
            F = fd.FunctionSpace(self.mesh, "CG", 1)
            rho_f = fd.Function(F, name = "rho_f")
            u_filter = fd.TrialFunction(F)
            v_filter = fd.TestFunction(F)

            A_filter = (self.rf ** 2 * inner(grad(u_filter), grad(v_filter)) * dx
                    + u_filter * v_filter * dx)
            L_filter = self.rho * v_filter * dx
            density_filter_problem = fd.LinearVariationalProblem(
                    A_filter, L_filter, rho_f)
            rhof_solve = fd.LinearVariationalSolver(density_filter_problem)

        elif (self.filter_scheme == "mixed"):
            # filter problem with mixed
            if self.is_quad:
                RT = fd.FunctionSpace(self.mesh, "RTCF", 1)
            else:
                RT = fd.FunctionSpace(self.mesh, "RT", 1)
            F = RT * D
            u_filter, g_filter = fd.TrialFunctions(F)
            v_filter, q_filter = fd.TestFunctions(F)
            gh = fd.Function(F)

            if args.dim == 2:
                zero = fd.as_vector([0., 0.])
            else:
                zero = fd.as_vector([0., 0., 0.])

            bc_filter = [fd.DirichletBC(F.sub(0), zero, "on_boundary")]
            A_filter = (
                    inner(u_filter, v_filter) * dx - g_filter * div(v_filter) * dx
                    + (self.rf ** 2) * div(u_filter) * q_filter * dx
                    + g_filter * q_filter * dx)
            L_filter = self.rho * q_filter * dx
            density_filter_problem = fd.LinearVariationalProblem(
                    A_filter, L_filter, gh, bcs=bc_filter)
            if args.dim == 2:
                filter_solver_parameters = {
                    "mat_type": "aij",
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                    }
            else:
                filter_solver_parameters = {
                    "mat_type": "nest",
                    "ksp_type": "gmres",
                    "ksp_rtol": 1e-4,
                    #"ksp_monitor": None,
                    #"ksp_converged_reason": None,
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_fact_type": "full",
                    "fieldsplit_0_ksp_type": "preonly",
                    "fieldsplit_0_pc_type": "bjacobi",
                    "fieldsplit_0_sub_pc_type": "ilu",
                    "fieldsplit_1_ksp_type": "preonly",
                    "pc_fieldsplit_schur_precondition": "selfp",
                    "fieldsplit_1_pc_type": "hypre",
                    "fieldsplit_1_pc_hypre_boomeramg": {
                    "strong_threshold": 0.7,
                    "coarsen_type": "HMIS",
                    "agg_nl": 3,
                    "interp_type": "ext+i",
                    "agg_num_paths": 5,
                    },
                }
            rhof_solve = fd.LinearVariationalSolver(density_filter_problem, solver_parameters = filter_solver_parameters)

        elif (self.filter_scheme == "TPFA"):
            rho_f = fd.Function(D, name = "rho_f")
            # filter problem with TPFA
            g_filter = fd.TrialFunction(D)
            v_filter = fd.TestFunction(D)

            x_center = fd.interpolate(self.x, D)
            y_center = fd.interpolate(self.y, D)
            if args.dim == 2:
                dh = ufl.sqrt(jump(x_center) ** 2 + jump(y_center) ** 2)
            else:
                z_center = fd.interpolate(self.z, D)
                dh = ufl.sqrt(jump(x_center) ** 2 + jump(y_center) ** 2 + jump(z_center) ** 2)


            A_filter = (self.rf ** 2 * jump(g_filter) / dh * jump(v_filter) * dS
                    + g_filter * v_filter * dx)
            L_filter = self.rho * v_filter * dx
            density_filter_problem = fd.LinearVariationalProblem(
                    A_filter, L_filter, rho_f)
            rhof_solve = fd.LinearVariationalSolver(density_filter_problem)

        rhof_solve.solve()
        if (self.filter_scheme == "mixed"):
            self.vel_filter, rho_f = gh.split()
            self.vel_filter.rename("grad_rhof")
            rho_f.rename("rho_f")

        rhof_control = fda.Control(rho_f)
        self.rho_f = rho_f
        if args.proj > 0:
            self.rho_f.interpolate(heaviside_projection(self.rho_f, args.proj, 0.5))

    def boundary_propagation(self):
        D = self.rho.function_space()
        if self.bd_prop_scheme == "CG":
            # boundary propagation problem with CG
            B = fd.FunctionSpace(self.mesh, "CG", 1)
            b_prop = fd.TrialFunction(B)
            v_prop = fd.TestFunction(B)
            self.beta_prop = fd.Function(B, name="beta")

            bc_prop = [fd.DirichletBC(B, 1., 3),
                       fd.DirichletBC(B, -1., 4)]
            A_prop = (self.rho_f * inner(grad(b_prop), grad(v_prop)) * dx
                      + (fd.Constant(1.) - self.rho_f) * b_prop * v_prop * dx)
            L_prop = fd.Constant(0.) * v_prop * dx
            boundary_propagation_problem = fd.LinearVariationalProblem(
               A_prop, L_prop, self.beta_prop, bcs=bc_prop)
            beta_solve = fd.LinearVariationalSolver(
               boundary_propagation_problem,
               solver_parameters=SuperCapacitor.solver_parameters)

        elif self.bd_prop_scheme == "mixed":
            # boundary propagation problem with mixed
            if self.is_quad:
                RT = fd.FunctionSpace(self.mesh, "RTCF", 1)
            else:
                RT = fd.FunctionSpace(self.mesh, "RT", 1)
            B = RT * D
            u_prop, b_prop = fd.TrialFunctions(B)
            v_prop, q_prop = fd.TestFunctions(B)
            bh = fd.Function(B)
            self.vel_prop, self.beta_prop = bh.split()

            if args.dim == 2:
                zero = fd.as_vector([0., 0.])
            else:
                zero = fd.as_vector([0., 0., 0.])
            bc_prop = [fd.DirichletBC(B.sub(0), zero, 1),
                       fd.DirichletBC(B.sub(0), zero, 2)]
            if args.dim == 3:
                bc_prop.extend([fd.DirichletBC(B.sub(0), zero, 5),
                       fd.DirichletBC(B.sub(0), zero, 6)])

            A_prop = (inner(1./(self.rho_f + 1e-8) * u_prop, v_prop) * dx
                    - b_prop * div(v_prop) * dx - div(u_prop) * q_prop * dx
                    - (fd.Constant(1.) - self.rho_f) * b_prop * q_prop * dx)
            L_prop = (- fd.Constant(1.) *
                      dot(v_prop, fd.FacetNormal(self.mesh)) * ds(3)
                      + fd.Constant(1.) *
                      dot(v_prop, fd.FacetNormal(self.mesh)) * ds(4))
            boundary_propagation_problem = fd.LinearVariationalProblem(
               A_prop, L_prop, bh, bcs=bc_prop)
            if args.dim == 2:
                beta_solver_parameters = {
                    "mat_type": "aij",
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                    }
            else:
                beta_solver_parameters = {
                    "mat_type": "nest",
                    "ksp_type": "gmres",
                    "ksp_rtol": 1e-4,
                    #"ksp_monitor": None,
                    #"ksp_converged_reason": None,
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_fact_type": "full",
                    "fieldsplit_0_ksp_type": "preonly",
                    "fieldsplit_0_pc_type": "bjacobi",
                    "fieldsplit_0_sub_pc_type": "ilu",
                    "fieldsplit_1_ksp_type": "preonly",
                    "pc_fieldsplit_schur_precondition": "selfp",
                    "fieldsplit_1_pc_type": "hypre",
                    "fieldsplit_1_pc_hypre_boomeramg": {
                    "strong_threshold": 0.7,
                    "coarsen_type": "HMIS",
                    "agg_nl": 3,
                    "interp_type": "ext+i",
                    "agg_num_paths": 5,
                    },
                }
            beta_solve = fd.LinearVariationalSolver(boundary_propagation_problem, solver_parameters = beta_solver_parameters)

        elif self.bd_prop_scheme == "TPFA":
            # boundary propagation problem with two point flux approximation
            b_prop = fd.TrialFunction(D)
            v_prop = fd.TestFunction(D)
            self.beta_prop = fd.Function(D, name="beta")

            x_center = fd.interpolate(self.x, D)
            y_center = fd.interpolate(self.y, D)
            if args.dim == 2:
                dh = ufl.sqrt(jump(x_center) ** 2 + jump(y_center) ** 2)
            else:
                z_center = fd.interpolate(self.z, D)
                dh = ufl.sqrt(jump(x_center) ** 2 + jump(y_center) ** 2 + jump(z_center) ** 2)
            dh_half = 0.5 * fd.assemble(v_prop * dx) / fd.FacetArea(self.mesh) #does this work in 3D?

            rho_facet = ufl.conditional(ufl.gt(ufl.avg(self.rho_f), 0.),
                    self.rho_f('+') * self.rho_f('-') / ufl.avg(self.rho_f), 0.)
            A_prop = (rho_facet * jump(b_prop) / dh * jump(v_prop) * dS
                      - (fd.Constant(1.) - self.rho_f) * b_prop * v_prop * dx
                      - 1. / dh_half * self.rho_f * b_prop * v_prop * ds(3)
                      - 1. / dh_half * self.rho_f * b_prop * v_prop * ds(4))
            L_prop = (- 1. / dh_half * self.rho_f *
                      fd.Constant(1.) * v_prop * ds(3)
                      + 1. / dh_half * self.rho_f *
                      fd.Constant(1.) * v_prop * ds(4))
            boundary_propagation_problem = fd.LinearVariationalProblem(
               A_prop, L_prop, self.beta_prop)
            beta_solve = fd.LinearVariationalSolver(boundary_propagation_problem)

        beta_solve.solve()
        if self.bd_prop_scheme == "CG":
            self.vel_prop = fd.Function(fd.VectorFunctionSpace(self.mesh, "DG", 0),
                    name="v_beta").interpolate(self.rho_f * grad(self.beta_prop))
            self.short_circ = fd.assemble(sc_weight * (1. - (1. - self.rho_f) ** 3) *
                inner(grad(self.beta_prop), grad(self.beta_prop)) * dx)

        elif self.bd_prop_scheme == "mixed":
            self.vel_prop, self.beta_prop = bh.split()
            self.vel_prop.rename("v_beta")
            self.beta_prop.rename("beta")
            self.short_circ = fd.assemble(sc_weight * self.rho_f *
                ((1.0 - abs(self.beta_prop)) ** 3) * dx)



    def solve_optimization(self, transform_control=None):
        global_counter = itertools.count()
        rhof_control = fda.Control(self.rho_f)
        beta_control = fda.Control(self.beta_prop)
        vbeta_control = fda.Control(self.vel_prop)
        u_control = fda.Control(self.u)
        loss_control = fda.Control(self.eloss_dist)
        redox_control = fda.Control(self.eredox_dist)
        capa_control = fda.Control(self.ecapa_dist)

        D = fd.FunctionSpace(self.mesh, "DG", 1)
        B = self.beta_prop.function_space()
        U = fd.VectorFunctionSpace(self.mesh, "DG", 0)
        rhof_vis = fd.Function(self.rho_f.function_space(), name="rho_f")
        beta_vis = fd.Function(B, name="beta")
        eloss_vis = fd.Function(D, name="E_loss")
        eredox_vis = fd.Function(D, name="E_redox")
        ecapa_vis = fd.Function(D, name="E_capac")

        opt_process = fd.File(args.output_dir+"/geometry.pvd")
        def deriv_cb(j, dj, gamma):
            iter = next(global_counter)
            if iter % 2 == 1:
                with fda.stop_annotating():
                    rhof_vis.assign(rhof_control.tape_value())
                    beta_vis.assign(beta_control.tape_value())
                    eloss_vis.assign(loss_control.tape_value())
                    eredox_vis.assign(redox_control.tape_value())
                    ecapa_vis.assign(capa_control.tape_value())
                    if args.csurf == 1 and args.conc == 1:
                        phi1c, phi1a, phi2, conc, csurf = u_control.tape_value().split()
                        conc.rename("conc")
                    elif args.csurf == 0 and args.conc == 1:
                        phi1c, phi1a, phi2, conc = u_control.tape_value().split()
                        conc.rename("conc")
                    else:
                        phi1c, phi1a, phi2 = u_control.tape_value().split()
                        conc = fd.Constant(1.0)
                    eflux = fd.interpolate(- self.kappa_hat * conc * grad(phi2), U)
                    if args.conc == 1:
                        eflux += fd.interpolate(- (self.t_plus0 / self.z_plus + self.t_minus0 / self.z_minus) *
                                self.kappa_hat * grad(conc), U)
                    beta_grad = fd.Function(self.vel_prop.function_space(), name="grad_beta").interpolate(
                            vbeta_control.tape_value() / rhof_control.tape_value())
                    eind = (beta_control.tape_value() + fd.Constant(1.)) / fd.Constant(2.)
                    cath_ind_vis = fd.Function(D, name="cath ind").interpolate(
                            (ufl.tanh(q_ind * 0.5) - ufl.tanh(q_ind * (eind - 0.5))) /
                            (ufl.tanh(q_ind * 0.5) - ufl.tanh(-q_ind * 0.5)))
                    ano_ind_vis = fd.Function(D, name="ano ind").interpolate(
                            (ufl.tanh(q_ind * (eind - 0.5)) - ufl.tanh(-q_ind * 0.5)) /
                            (ufl.tanh(q_ind * 0.5) - ufl.tanh(-q_ind * 0.5)))
                    phi1c.rename("phi_1_cathode")
                    phi1a.rename("phi_1_anode")
                    phi2.rename("phi_2")
                    eflux.rename("grad_phi_2")
                    if args.conc == 1:
                        opt_process.write(rhof_vis, beta_vis, beta_grad, eloss_vis, eredox_vis, ecapa_vis,
                                phi1c, phi1a, phi2, conc, cath_ind_vis, ano_ind_vis, eflux, mode="a")
                    else:
                        opt_process.write(rhof_vis, beta_vis, beta_grad, eloss_vis, eredox_vis, ecapa_vis,
                                phi1c, phi1a, phi2, cath_ind_vis, ano_ind_vis, eflux, mode="a")
            if iter % (args.max_iter - args.chk_init) == 0 and iter > 0:
                os.kill(os.getpid(), signal.SIGUSR1)
            return dj

        rho_control = fda.Control(self.rho)
        
        # choose an optimization strategy (objective and constraint)
        if args.opt_strat==0:
            # Maximize energy stored with constraint on Ohmic loss
            # continuation on conductivity penalization from pinit to 1.0
            parprint("Opt strategy: Max energy stored with constraint on Ohmic loss")
            Jhat = fda.ReducedFunctional(self.ebalance, rho_control,
                    derivative_cb_post=deriv_cb)

            cval = self.efficiency * engy_cval
            effi_control = fda.Control(self.efficiency)
            Fhat = fda.ReducedFunctional(self.efficiency, rho_control)
            effi_constraint = ReducedInequality(Fhat, cval, effi_control)

            loop = args.chk_init
            p_cond_control = fda.Control(p_cond_opt)
            SuperCapacitor.mma_parameters["maximum_iterations"] = args.loop_init
            if (args.chk_init != 0):
                SuperCapacitor.mma_parameters["restart_file"] = "./checkpoint_iter_"+str(args.chk_init)+".h5"
            if args.opt_cycle != 0:
                p_step = (1.0 - args.p_init) / args.opt_cycle
                loop_step = (args.max_iter - args.loop_init) / args.opt_cycle
            else:
                p_step = 0.
                loop_step = 0.

            for n in range(args.opt_cycle + 1):
                optimization_problem = fda.MinimizationProblem(Jhat, bounds=(0.0, 1.0),
                                                               constraints=[effi_constraint])
                self.opt_solver = MMASolver(optimization_problem,
                        parameters=SuperCapacitor.mma_parameters)
                if transform_control is not None:
                    self.opt_solver.transform_control = transform_control

                results = self.opt_solver.solve(loop=loop)

                rho_opt = results["control"]
                loop = results["loop"]
                self.rho.assign(rho_opt)

                p_cond_control.data().assign(p_cond_control.data().values()[0] + p_step)
                if (n + 1 == args.opt_cycle):
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step + 50
                else:
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step

        elif args.opt_strat==1:
            # Maximize energy stored with constraint on Ohmic loss
            # continuation on delta
            parprint("Opt strategy: Max energy stored with constraint on Ohmic loss")
            Jhat = fda.ReducedFunctional(self.ebalance, rho_control,
                    derivative_cb_post=deriv_cb)

            cval = self.efficiency * engy_cval
            effi_control = fda.Control(self.efficiency)
            Fhat = fda.ReducedFunctional(self.efficiency, rho_control)
            effi_constraint = ReducedInequality(Fhat, cval, effi_control)

            loop = 0
            p_cond_control = fda.Control(p_cond_opt)
            delta_r_control = fda.Control(self.delta_redox)
            delta_c_control = fda.Control(self.delta_capac)
            SuperCapacitor.mma_parameters["maximum_iterations"] = args.loop_init
            if args.opt_cycle != 0:
                p_step = (1.0 - args.p_init) / args.opt_cycle
                loop_step = (args.max_iter - args.loop_init) / args.opt_cycle
            else:
                p_step = 0.
                loop_step = 0.

            for n in range(args.opt_cycle + 1):
                parprint(delta_r_control.data().values())
                parprint(delta_c_control.data().values())
                optimization_problem = fda.MinimizationProblem(Jhat, bounds=(0.0, 1.0),
                                                               constraints=[effi_constraint])
                self.opt_solver = MMASolver(optimization_problem,
                        parameters=SuperCapacitor.mma_parameters)

                results = self.opt_solver.solve(loop=loop)
                rho_opt = results["control"]
                loop = results["loop"]
                self.rho.assign(rho_opt)

                p_cond_control.data().assign(p_cond_control.data().values()[0] + p_step)
                delta_r_control.data().assign(self.delta * self.beta)
                delta_c_control.data().assign(self.delta * (1.0 - self.beta))
                self.delta_redox.assign(self.delta * self.beta)
                self.delta_capac.assign(self.delta * (1.0 - self.beta))

                if (n + 1 == args.opt_cycle):
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step + 50
                else:
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step

                cval = cval * 2.0
                effi_constraint = ReducedInequality(Fhat, cval, effi_control)

        if args.opt_strat==2:
            # Maximize energy stored with constraint on Ohmic loss
            # continuation on modified bruggeman scaling from 0.02 to mod_brugg
            parprint("Opt strategy: Max energy stored with constraint on Ohmic loss")
            Jhat = fda.ReducedFunctional(self.ebalance, rho_control,
                    derivative_cb_post=deriv_cb)

            cval = self.efficiency * engy_cval
            effi_control = fda.Control(self.efficiency)
            Fhat = fda.ReducedFunctional(self.efficiency, rho_control)
            effi_constraint = ReducedInequality(Fhat, cval, effi_control)

            loop = 0
            p_brugg_control = fda.Control(p_mod_brugg)
            SuperCapacitor.mma_parameters["maximum_iterations"] = args.loop_init
            if args.opt_cycle != 0:
                p_step = (args.mod_brugg - 0.02) / args.opt_cycle
                loop_step = (args.max_iter - args.loop_init) / args.opt_cycle
            else:
                p_step = 0.
                loop_step = 0.

            for n in range(args.opt_cycle + 1):
                optimization_problem = fda.MinimizationProblem(Jhat, bounds=(0.0, 1.0),
                                                               constraints=[effi_constraint])
                self.opt_solver = MMASolver(optimization_problem,
                        parameters=SuperCapacitor.mma_parameters)

                results = self.opt_solver.solve(loop=loop)
                rho_opt = results["control"]
                loop = results["loop"]
                self.rho.assign(rho_opt)

                p_brugg_control.data().assign(p_brugg_control.data().values()[0] + p_step)
                if (n + 1 == args.opt_cycle):
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step + 50
                else:
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step
            p_mod_brugg.assign(args.mod_brugg)

        if args.opt_strat==3:
            # Maximize energy stored with constraint on Ohmic loss
            # continuation on modified bruggeman scaling from 0.02 to mod_brugg
            # and also continuation on delta
            parprint("Opt strategy: Max energy stored with constraint on Ohmic loss")
            Jhat = fda.ReducedFunctional(self.ebalance, rho_control,
                    derivative_cb_post=deriv_cb)

            cval = self.efficiency * engy_cval
            effi_control = fda.Control(self.efficiency)
            Fhat = fda.ReducedFunctional(self.efficiency, rho_control)
            effi_constraint = ReducedInequality(Fhat, cval, effi_control)

            loop = 0
            p_brugg_control = fda.Control(p_mod_brugg)
            delta_r_control = fda.Control(self.delta_redox)
            delta_c_control = fda.Control(self.delta_capac)
            SuperCapacitor.mma_parameters["maximum_iterations"] = args.loop_init
            if args.opt_cycle != 0:
                p_step = (args.mod_brugg - 0.02) / args.opt_cycle
                loop_step = (args.max_iter - args.loop_init) / args.opt_cycle
            else:
                p_step = 0.
                loop_step = 0.

            for n in range(args.opt_cycle + 1):
                optimization_problem = fda.MinimizationProblem(Jhat, bounds=(0.0, 1.0),
                                                               constraints=[effi_constraint])
                self.opt_solver = MMASolver(optimization_problem,
                        parameters=SuperCapacitor.mma_parameters)

                results = self.opt_solver.solve(loop=loop)
                rho_opt = results["control"]
                loop = results["loop"]
                self.rho.assign(rho_opt)

                p_brugg_control.data().assign(p_brugg_control.data().values()[0] + p_step)
                delta_r_control.data().assign(self.delta * self.beta)
                delta_c_control.data().assign(self.delta * (1.0 - self.beta))
                self.delta_redox.assign(self.delta * self.beta)
                self.delta_capac.assign(self.delta * (1.0 - self.beta))
                if (n + 1 == args.opt_cycle):
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step + 50
                else:
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step

                cval = cval * 2.0
                effi_constraint = ReducedInequality(Fhat, cval, effi_control)
            p_mod_brugg.assign(args.mod_brugg)

        if args.opt_strat==4:
            # Continuation on increasing cval to enlarge viable solution space for small delta cases
            parprint("Opt strategy: Max energy stored with constraint on Ohmic loss")
            Jhat = fda.ReducedFunctional(self.ebalance, rho_control,
                    derivative_cb_post=deriv_cb)

            ival = self.eloss / self.engyin
            cval = self.efficiency * engy_cval
            effi_control = fda.Control(self.efficiency)
            Fhat = fda.ReducedFunctional(self.efficiency, rho_control)
            effi_constraint = ReducedInequality(Fhat, cval, effi_control)

            loop = 0
            p_cond_control = fda.Control(p_cond_opt)
            SuperCapacitor.mma_parameters["maximum_iterations"] = args.loop_init
            if args.opt_cycle != 0:
                loop_step = (args.max_iter - args.loop_init) / args.opt_cycle
            else:
                loop_step = 0.

            for n in range(args.opt_cycle + 1):
                parprint(p_cond_control.data().values())
                optimization_problem = fda.MinimizationProblem(Jhat, bounds=(0.0, 1.0),
                                                               constraints=[effi_constraint])
                self.opt_solver = MMASolver(optimization_problem,
                        parameters=SuperCapacitor.mma_parameters)

                results = self.opt_solver.solve(loop=loop)
                rho_opt = results["control"]
                loop = results["loop"]
                self.rho.assign(rho_opt)

                cval = ival
                effi_constraint = ReducedInequality(Fhat, cval, effi_control)
                if (n + 1 == args.opt_cycle):
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step + 50
                else:
                    SuperCapacitor.mma_parameters["maximum_iterations"] += loop_step

        elif args.opt_strat==5:
            # Continuation on scaling of energy storage and ohmic loss
            # as postprocessing step for large delta case to produce binary design
            parprint("Opt strategy: Max energy stored with constraint on Ohmic loss")
            Jhat = fda.ReducedFunctional(self.ebalance, rho_control,
                    derivative_cb_post=deriv_cb)

            cval = self.efficiency * engy_cval
            effi_control = fda.Control(self.efficiency)
            Fhat = fda.ReducedFunctional(self.efficiency, rho_control)
            effi_constraint = ReducedInequality(Fhat, cval, effi_control)

            loop = 0
            p_cond_control = fda.Control(p_cond_opt)
            scale_control = fda.Control(beta_scale)
            delta_r_control = fda.Control(self.delta_redox)
            delta_c_control = fda.Control(self.delta_capac)
            SuperCapacitor.mma_parameters["maximum_iterations"] = args.loop_init

            parprint(delta_r_control.data().values())
            parprint(delta_c_control.data().values())
            optimization_problem = fda.MinimizationProblem(Jhat, bounds=(0.0, 1.0),
                                                           constraints=[effi_constraint])
            self.opt_solver = MMASolver(optimization_problem,
                    parameters=SuperCapacitor.mma_parameters)

            results = self.opt_solver.solve(loop=loop)
            rho_opt = results["control"]
            loop = results["loop"]
            self.rho.assign(rho_opt)

            delta_r_control.data().assign(self.delta * self.beta)
            delta_c_control.data().assign(self.delta * (1.0 - self.beta))
            self.delta_redox.assign(self.delta * self.beta)
            self.delta_capac.assign(self.delta * (1.0 - self.beta))
            SuperCapacitor.mma_parameters["maximum_iterations"] = 150

            cval = cval * 2.0
            effi_constraint = ReducedInequality(Fhat, cval, effi_control)

            parprint(delta_r_control.data().values())
            parprint(delta_c_control.data().values())
            optimization_problem = fda.MinimizationProblem(Jhat, bounds=(0.0, 1.0),
                                                           constraints=[effi_constraint])
            self.opt_solver = MMASolver(optimization_problem,
                    parameters=SuperCapacitor.mma_parameters)

            results = self.opt_solver.solve(loop=loop)
            rho_opt = results["control"]
            loop = results["loop"]
            self.rho.assign(rho_opt)

            scale_control.data().assign(1.0)
            SuperCapacitor.mma_parameters["maximum_iterations"] = args.max_iter + 50
            optimization_problem = fda.MinimizationProblem(Jhat, bounds=(0.0, 1.0),
                                                           constraints=[effi_constraint])
            self.opt_solver = MMASolver(optimization_problem,
                    parameters=SuperCapacitor.mma_parameters)

            results = self.opt_solver.solve(loop=loop)
            rho_opt = results["control"]
            loop = results["loop"]
            self.rho.assign(rho_opt)


        with fd.CheckpointFile(args.output_dir+"/optimized_rho.h5", 'w') as chkptfile:
            chkptfile.save_mesh(self.mesh)
            chkptfile.save_function(self.rho)

        self.rho_f.interpolate(ufl.conditional(self.rho_f < 0.5, fd.Constant(0.0), fd.Constant(1.0)))
        with fd.CheckpointFile(args.output_dir+"/optimized_proj_rho.h5", 'w') as chkptfile:
            chkptfile.save_mesh(self.mesh)
            chkptfile.save_function(self.rho_f)


    def save_stats(self):

        solver = self.solver
        comm = COMM_WORLD

        snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
        ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        jac = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()

        snes_time = comm.allreduce(snes["time"], op=MPI.SUM) / comm.size
        jac_time = comm.allreduce(jac["time"], op=MPI.SUM) / comm.size
        residual_time = comm.allreduce(
            residual["time"], op=MPI.SUM) / comm.size
        ksp_time = comm.allreduce(ksp["time"], op=MPI.SUM) / comm.size
        pcsetup_time = comm.allreduce(
            pcsetup["time"], op=MPI.SUM) / comm.size
        pcapply_time = comm.allreduce(
            pcapply["time"], op=MPI.SUM) / comm.size

        newton_its = solver.snes.getIterationNumber()
        ksp_its = solver.snes.getLinearSolveIterations()
        num_cells = comm.allreduce(self.mesh.cell_set.size, op=MPI.SUM)

        stats = os.path.abspath('data.csv')
        if COMM_WORLD.rank == 0:
            snes_history, linear_its = solver.snes.getConvergenceHistory()
            ksp_history = solver.snes.ksp.getConvergenceHistory()

            data = {
                "num_processes": comm.size,
                "num_cells": num_cells,
                "dimension": args.dim,
                "dofs": self.u.dof_dset.layout_vec.getSize(),
                "snes_its": newton_its,
                "ksp_its": ksp_its,
                "SNESSolve": snes_time,
                "KSPSolve": ksp_time,
                "PCSetUp": pcsetup_time,
                "PCApply": pcapply_time,
                "JacobianEval": jac_time,
                "FunctionEval": residual_time,
                "mesh_size": args.Nx * 2,
            }

            if not os.path.exists(os.path.dirname(stats)):
                os.makedirs(os.path.dirname(stats))

            if False:
                mode = "w"
                header = True
            else:
                mode = "a"
                header = not os.path.exists(stats)

            df = pandas.DataFrame(data, index=[0])
            df.to_csv(stats, index=False, mode=mode, header=header)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank =comm.Get_rank()
    if rank == 0:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    comm.Barrier()

    if args.dimless_entry == 1:
        two_electrodes = SuperCapacitor(args.input_dir+"/parameter_dimless.in")
    else:
        two_electrodes = SuperCapacitor(args.input_dir+"/parameter.in", dimless_entry=False)

    two_electrodes.setup_problem()
    if args.forward == 0:
        initial_solution = fd.File(args.output_dir+"/initial_solution.pvd")
        two_electrodes.forward(initial_solution)
        if args.force_break == 1:
            def transform_control(rho):
                electrode_ind = (two_electrodes.beta_prop + fd.Constant(1.)) / fd.Constant(2.)
                cath_ind = ((ufl.tanh(q_ind * fd.Constant(0.5 - offset)) - ufl.tanh(q_ind * (electrode_ind - fd.Constant(0.5 - offset)))) /
                        (ufl.tanh(q_ind * fd.Constant(0.5 - offset)) - ufl.tanh(-q_ind * fd.Constant(0.5 + offset))))
                ano_ind = ((ufl.tanh(q_ind * (electrode_ind - fd.Constant(0.5 + offset))) - ufl.tanh(-q_ind * fd.Constant(0.5 + offset))) /
                        (ufl.tanh(q_ind * fd.Constant(0.5 + offset)) - ufl.tanh(-q_ind * fd.Constant(0.5 - offset))))
                rho.interpolate(rho * (cath_ind + ano_ind))
        else:
            transform_control = None
        two_electrodes.solve_optimization(transform_control=transform_control)
        final_solution = fd.File(args.output_dir+"/final_solution.pvd")
        p_cond_opt.assign(1.5)
        p_engy_opt.assign(1.)
        p_mod_brugg.assign(args.mod_brugg)
        # clear cache
        if args.dim == 3:
            TSFCKernel._cache.clear()
            GlobalKernel._cache.clear()
            gc.collect()
            petsc4py.PETSc.garbage_cleanup(two_electrodes.mesh._comm)
            petsc4py.PETSc.garbage_cleanup(two_electrodes.mesh.comm)
        two_electrodes.forward(final_solution)
    else:
        final_solution = fd.File(args.output_dir+"/final_solution.pvd")
        p_cond_opt.assign(1.5)
        p_engy_opt.assign(1.)
        p_mod_brugg.assign(args.mod_brugg)
        two_electrodes.forward(final_solution)
        #two_electrodes.save_stats()
