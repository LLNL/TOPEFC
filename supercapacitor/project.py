import flow
from flow import FlowProject
from init import parameters
import platform
import re
from matplotlib import pyplot as plt
from signac import get_project

project = get_project()
xi_range = [0.01, 0.05, 0.1, 0.5, 1.0]


@FlowProject.label
def check_iterations(job):
    filename = f"geometry.pvd"
    return job.isfile(filename)


@FlowProject.label
def check_h5_design(job):
    return job.isfile("final_design_uniform")


@FlowProject.label
def check_output(job):
    return job.isfile("output.txt")

@FlowProject.label
def check_input_file(job):
    return job.isfile("parameter_dimless.in")

@FlowProject.operation
@flow.cmd
@FlowProject.post(check_input_file)
def make_input_file(job):
    # copy parameter_dimless.in to job.ws
    original = "parameter_dimless.in"
    new = job.ws + "/" + original
    input_file = open(original, "r")
    data = input_file.readlines()
    i = 0
    for line in data:
        var_name, var_value = line.split("=")
        if var_name.strip() in parameters:  
            data[i] =  var_name + '=\t' + str(job.sp.get(var_name.strip())) + '\n'
        i += 1
    input_file = open(new, "w")
    input_file.writelines(data)

    cp_cmd = "cp supercapacitor.py " + job.ws + "/"
    return cp_cmd

@FlowProject.operation
@flow.cmd
@FlowProject.pre(check_input_file)
@FlowProject.post(check_output)
@FlowProject.post(check_iterations)
# @FlowProject.post(check_h5_design)
def launch_opti(job):
    # program = "supercapacitor.py"
    program = job.ws + "/supercapacitor.py"
    param_flags = [
        "--{0} {1} ".format(param, job.sp.get(param)) for param in parameters
    ]
    output = job.ws + "/output.txt"
    plat = platform.system()
    proc = 32 if job.sp.get("dim", None) == 3 else 4
    if plat == "Linux":
        simulation = "srun -n {4} --output={0} python3 {3} {1} --output_dir {2} --input_dir {2}".format(
            output, "".join(param_flags), job.ws, program, proc
        )
    else:
        simulation = "python3 {3} {0} --output_dir {1}/ > {2}".format(
            "".join(param_flags), job.ws, output, program
        )
    print(simulation)
    return simulation


@FlowProject.label
def check_figures(job):
    return all(
        job.isfile(file)
        for file in ["convergence_plots_0.svg", "convergence_plots_1.svg"]
    )


@FlowProject.operation
@FlowProject.pre(check_iterations)
@FlowProject.post(check_figures)
def plot_history(job):
    from read_history import parse_history
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    (
        iter_history_current,
        obj_iteration_history,
        constr_iteration_history,
    ) = parse_history(f"{job.ws}/output.txt")
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.plot(iter_history_current[:300], obj_iteration_history[:300])
    ax1.set_xlabel("Number of iterations", fontsize=18)
    ax1.set_ylabel("Cost function", fontsize=18)
    ax1.set_xscale('log')

    ax2.plot(iter_history_current[:300], constr_iteration_history[:300])
    ax2.set_xlabel("Number of iterations", fontsize=18)
    ax2.set_ylabel("Constraint function", fontsize=18)

    for i, figura in enumerate([fig1, fig2]):
        plt.figure(figura.number)
        plt.savefig(
            f"{job.ws}/convergence_plots_{i}.svg",
            dpi=1600,
            orientation="portrait",
            format=None,
            transparent=True,
            bbox_inches="tight",
        )


@FlowProject.label
def check_final_cost(job):
    return job.isfile("results.txt")


@FlowProject.operation
@FlowProject.pre(check_iterations)
@FlowProject.post(check_final_cost)
def write_optimal_information(job):

    if job.sp.get("opt_strat", None) == 0:
        # Read value of the constraint
        eng_number = "-?\d+\.?\d*(?:e-?\d+)?"
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "Value: ({0}), Constraint ({0})".format(eng_number), sim_output.read()
            )
        lossratio_acquired = float(results[-1][0])
        lossratio_required = float(results[-1][1])

        # Read value of energy loss
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "ohmic loss = ({0})".format(eng_number), sim_output.read()
            )
        final_energy_loss = results[-1]
        # Read value of energy stored
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "energy stored = ({0})".format(eng_number), sim_output.read()
            )
        final_energy_stored = results[-1]
        # Read value of energy input
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "energy input = ({0})".format(eng_number), sim_output.read()
            )
        final_energy_input = results[-1]

        job.doc["final_energy_loss"] = final_energy_loss
        job.doc["energy_stored"] = final_energy_stored
        job.doc["energy_input"] = final_energy_input

        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "redox energy stored = ({0})".format(eng_number), sim_output.read()
            )
        redox_energy = results[-1]
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "capacitance energy stored = ({0})".format(eng_number), sim_output.read()
            )
        capa_energy = results[-1]
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "intermediate material = ({0})".format(eng_number), sim_output.read()
            )
        intermediate_material = results[-1]

        with open("{0}/results.txt".format(job.ws), "w") as results:
            results.write(f"delta: {job.sp['delta']}\t zeta: {job.sp['zeta']}\n")
            results.write(
                    f"Redox energy stored: {redox_energy}\nCapacitance energy stored: {capa_energy}\nEnergy stored: {final_energy_stored}\n")
            results.write(
                    f"Ohmic loss: {final_energy_loss}\nEnergy input: {final_energy_input}\n")
            results.write(
                    f"Loss ratio acquired: {lossratio_acquired}\nLoss ratio required: {lossratio_required}\nIntermediate material: {intermediate_material}\n")

    elif job.sp.get("opt_strat", None) == 1:
        # Read value of the constraint
        eng_number = "-?\d+\.?\d*(?:e-?\d+)?"
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "Value: ({0}), Constraint ({0})".format(eng_number), sim_output.read()
            )
        energy_stored = 1.0 / float(results[-1][0])
        energy_required = 1.0 / float(results[-1][1])

        # Read value of the cost function
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "obj: ({0}) g\[0\]: ({0})".format(eng_number), sim_output.read()
            )
        final_result = results[-1]

        # Read value of initial resistive energy so we can get the real energy loss
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "ohmic loss = ({0})".format(eng_number), sim_output.read()
            )
        initial_energy_loss = results[0]
        final_energy_loss = float(final_result[0]) * float(initial_energy_loss)

        job.doc["final_energy_loss"] = final_energy_loss
        job.doc["energy_stored"] = energy_stored

        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "redox energy stored = ({0})".format(eng_number), sim_output.read()
            )
        redox_energy = results[-1]
        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "capacitance energy stored = ({0})".format(eng_number), sim_output.read()
            )
        capa_energy = results[-1]

        with open("{0}/output.txt".format(job.ws), "r") as sim_output:
            results = re.findall(
                "intermediate material = ({0})".format(eng_number), sim_output.read()
            )
        intermediate_material = results[-1]

        # Write both
        with open("{0}/results.txt".format(job.ws), "w") as results:
            # results.write(f"pinit: {job.sp['p_init']}\t energy_constraint: {job.sp['engy_cval']}\n")
            results.write(f"energy_constraint: {job.sp['engy_cval']}\t resolution: {job.sp['Nx']}\n")
            results.write(
                    f"Cost function: {final_energy_loss}\nRedox energy stored: {redox_energy}\nCapacitance energy stored: {capa_energy}\n")
            results.write(
                    f"Energy stored: {energy_stored}\nEnergy required: {energy_required}\nIntermediate material: {intermediate_material}\n")


@FlowProject.label
def check_design(job):
    return job.isfile(f"{screenshot_name(job)}.png")


def screenshot_name(job):
    # return f"design_ecval_{job.sp['engy_cval']}_Nx_{job.sp['Nx']}"
    return f"brugg_{job.sp['mod_brugg']}_delta_{job.sp['delta']}_gamma_{job.sp['beta']}_lambda_{job.sp['lambda']}"


@FlowProject.operation
@flow.cmd
@FlowProject.pre(check_iterations)
@FlowProject.post(check_design)
def post_process_design(job):
    parameters = "".join([key + " " + f"{job.sp[key]}" + "\n" for key in job.sp.keys()])

    plat = platform.system()
    return (
        "srun pvpython screenshot_design.py \
                --parameters '{0}' \
                --filename {1} \
                --results_dir {2} && \
                convert {2}/{1}.png -trim {2}/{1}.png".format(
            parameters, screenshot_name(job), job.ws
        )
        if plat == "Linux"
        else "/Applications/ParaView-5.8.0.app/Contents/bin/pvpython screenshot_design.py \
                --parameters '{0}' \
                --filename {1} \
                --results_dir {2} && \
                convert {2}/{1}.png -trim {2}/{1}.png".format(
            parameters, screenshot_name(job), job.ws
        )
    )


@FlowProject.label
def files_scan(job):
    files_scan = [f"xi_{xi}.npz" for xi in xi_range]
    return all(job.isfile(file) for file in files_scan)


@FlowProject.operation
@flow.cmd
@FlowProject.pre(check_h5_design)
@FlowProject.post(files_scan)
def run_simulation_design(job):
    program = "supercap.py"
    simulation = ""
    parameters_sim = dict(job.sp.items())
    parameters_sim["initial_design"] = "not_uniform"

    for xi in xi_range:
        parameters_sim["xi"] = xi
        param_flags = [
            "--{0} {1} ".format(key, param) for key, param in parameters_sim.items()
        ]
        print(param_flags)
        output = job.ws + "/output_simulation.txt"
        plat = platform.system()
        proc = 60 if job.sp.get("dim", None) == 3 else 1
        if plat == "Linux":
            simulation += "srun -n {4} --output={0} python3 {3} {1} --output_dir {2} --forward\n".format(
                output, "".join(param_flags), job.ws, program, proc
            )
        else:
            simulation += "python3 {3} {0} --output_dir {1} --forward / > {2}\n".format(
                "".join(param_flags), job.ws, output, program
            )
    return simulation


if __name__ == "__main__":
    FlowProject().main()
