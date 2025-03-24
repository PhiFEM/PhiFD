import numpy as np


# Function used to write in the outputs files
def output_latex(f, A, B):
    for i in range(len(A)):
        f.write("(")
        f.write(str(A[i]))
        f.write(",")
        f.write(str(B[i]))
        f.write(")\n")
    # f.write("\n")


test_case = "circle_cos"
f = open(f"output_{test_case}.txt", "w")
f.write("\\begin{tikzpicture}")
f.write(
    "\\begin{loglogaxis}[name = ax1, width = .45\\textwidth, ylabel = Computation time (s),\n"
    + "xlabel = $L^2$ relative error,\n"
    + "xmin = 0.000001, xmax = 0.000012, \n"
    + "legend style = {at={(1,1)},anchor=south east, legend columns =1,\n"
    + " /tikz/column 2/.style={column sep = 10pt}}]\n",
)

Ns = [
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
]
N_multigrid = [
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
]
markers = ["*", "x", "+"]
colors = ["blue", "red", "orange"]
for i, method in enumerate(["direct", "iterative", "multigrid_interpolate_phi"]):
    times = np.load(f"./results_{test_case}/list_times_{method}_{test_case}.npy")
    error = np.load(f"./results_{test_case}/list_L2_{method}_{test_case}.npy")
    if method == "direct" or method == "iterative":
        NN = Ns
        pos = "right"
    else:
        NN = N_multigrid
        pos = "below left"
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
f.write("\legend{Direct, Iterative, Multigrid} \n")
f.write("\end{loglogaxis}\n")
f.write("\end{tikzpicture}")
f.close()
