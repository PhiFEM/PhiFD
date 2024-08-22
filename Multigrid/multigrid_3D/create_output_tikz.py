import numpy as np
import matplotlib.pyplot as plt


# Function used to write in the outputs files
def output_latex(f, A, B):
    for i in range(len(A)):
        f.write("(")
        f.write(str(A[i]))
        f.write(",")
        f.write(str(B[i]))
        f.write(")\n")
    # f.write("\n")


test_case = "sphere_cos"
f = open(f"output_{test_case}.txt", "w")
f.write("\\begin{tikzpicture}")
f.write(
    "\\begin{loglogaxis}[name = ax1, width = .45\\textwidth, ylabel = Computation time (s),\n"
    + "xlabel = $L^2$ relative error,\n"
    + "xmin = 0.000065, xmax = 0.002, \n"
    + "legend style = {at={(1,1)},anchor=south east, legend columns =1,\n"
    + " /tikz/column 2/.style={column sep = 10pt}}]\n",
)

Ns = [100, 110, 120, 140, 160, 180]
N_multigrid = [40, 50, 60, 70, 80, 90, 100]
markers = ["x", "+"]
colors = ["red", "orange"]
for i, method in enumerate(["reference", "multigrid_interpolate_phi"]):
    times = np.load(f"./results_{test_case}/list_times_{method}_{test_case}.npy")
    error = np.load(f"./results_{test_case}/list_L2_{method}_{test_case}.npy")
    if method == "reference":
        NN = Ns
        pos = "right"
    else:
        NN = N_multigrid
        pos = "below"
    f.write(f"\\addplot[mark={markers[i]}, {colors[i]}] coordinates " + "{\n")
    output_latex(f, error, times)
    f.write("};\n\n")
    for j in range(len(NN)):
        f.write(
            f"\\node [{pos}] at (axis cs:  {error[j]},  {times[j]})"
            + "{\\tiny"
            + f"$N_0=${NN[j]}"
            + "};\n"
        )
f.write("\legend{Reference iterative, Multigrid} \n")
f.write("\end{loglogaxis}\n")
f.write("\end{tikzpicture}")
f.close()
