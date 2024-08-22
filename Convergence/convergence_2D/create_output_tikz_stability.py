import numpy as np


# Function used to write in the outputs files
def output_latex(file, A, B):
    for i in range(len(A)):
        file.write("(")
        file.write(str(A[i]))
        file.write(",")
        file.write(str(B[i]))
        file.write(")\n")
    # file.write("\n")


res_phiFD = np.load("./full_res_with_cond_phiFD1.npy")
res_phiFD2 = np.load("./full_res_with_cond_phiFD2.npy")

sizes = res_phiFD[0, :, 0, 0]
Gamma = [0.001, 0.01, 0.1, 1.0, 10.0, 20.0]
Sigma = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
index_sigma_1 = 1
index_gamma_1 = 3
index_sigma_2 = 1
index_gamma_2 = 4

L2_error_fixed_sigma = res_phiFD[1, :, :, index_sigma_1]
L2_error_fixed_gamma = res_phiFD[1, :, index_gamma_1, :]

Cond_fixed_sigma = res_phiFD[-1, :, :, index_sigma_1]
Cond_fixed_gamma = res_phiFD[-1, :, index_gamma_1, :]

L2_error_fixed_sigma_2 = res_phiFD2[1, :, :, index_sigma_2]
L2_error_fixed_gamma_2 = res_phiFD2[1, :, index_gamma_2, :]

Cond_fixed_sigma_2 = res_phiFD2[-1, :, :, index_sigma_2]
Cond_fixed_gamma_2 = res_phiFD2[-1, :, index_gamma_2, :]


markers = ["*", "x", "+"]
colors = ["darkviolet", "cardinal", "ForestGreen"]

f = open(f"output_stability.txt", "w")
f.write("\\begin{tikzpicture}")
f.write(
    "\\begin{loglogaxis}[name = ax1, width = .45\\textwidth, ylabel = $L^2$ relative error,\n"
    + "xlabel = $\\sigma$,\n"
    + "legend style = {at={(1,1)},anchor=south east, legend columns =2,\n"
    + " /tikz/column 2/.style={column sep = 10pt}}]\n",
)

for i, h in enumerate(sizes):
    f.write(f"\\addplot[mark={markers[i]}, {colors[i]}] coordinates " + "{\n")
    output_latex(f, Sigma, L2_error_fixed_gamma[i])
    f.write("};\n\n")
    f.write(f"\\addlegendentry" + r"{$\phi$-FD $h=$" + f"{sizes[i]:.2f}" + "}\n")
    f.write(
        f"\\addplot[mark={markers[i]}, {colors[i]}, densely dashed] coordinates "
        + "{\n"
    )
    output_latex(f, Sigma, L2_error_fixed_gamma_2[i])
    f.write("};\n\n")
    f.write(f"\\addlegendentry" + r"{$\phi$-FD2 $h=$" + f"{sizes[i]:.2f}" + "}\n")
f.write("\end{loglogaxis}\n")
f.write("\end{tikzpicture}")
f.write("\\quad")
f.write("\\begin{tikzpicture}")
f.write(
    "\\begin{loglogaxis}[name = ax1, width = .45\\textwidth, ylabel = Condition number,\n"
    + "xlabel = $\\sigma$,\n"
    + "legend style = {at={(1,1)},anchor=south east, legend columns =2,\n"
    + " /tikz/column 2/.style={column sep = 10pt}}]\n",
)

for i, h in enumerate(sizes):
    f.write(f"\\addplot[mark={markers[i]}, {colors[i]}] coordinates " + "{\n")
    output_latex(f, Sigma, Cond_fixed_gamma[i])
    f.write("};\n\n")
    f.write(f"\\addlegendentry" + r"{$\phi$-FD $h=$" + f"{sizes[i]:.2f}" + "}\n")
    f.write(
        f"\\addplot[mark={markers[i]}, {colors[i]}, densely dashed] coordinates "
        + "{\n"
    )
    output_latex(f, Sigma, Cond_fixed_gamma_2[i])
    f.write("};\n\n")
    f.write(f"\\addlegendentry" + r"{$\phi$-FD2 $h=$" + f"{sizes[i]:.2f}" + "}\n")
f.write("\end{loglogaxis}\n")
f.write("\end{tikzpicture}")
f.write("\\\\")
f.write("\\begin{tikzpicture}")
f.write(
    "\\begin{loglogaxis}[name = ax1, width = .45\\textwidth, ylabel = $L^2$ relative error,\n"
    + "xlabel = $\\gamma$,\n"
    + "legend style = {at={(1,1)},anchor=south east, legend columns =2,\n"
    + " /tikz/column 2/.style={column sep = 10pt}}]\n",
)

for i, h in enumerate(sizes):
    f.write(f"\\addplot[mark={markers[i]}, {colors[i]}] coordinates " + "{\n")
    output_latex(f, Gamma, L2_error_fixed_sigma[i])
    f.write("};\n\n")
    f.write(f"\\addlegendentry" + r"{$\phi$-FD $h=$" + f"{sizes[i]:.2f}" + "}\n")
    f.write(
        f"\\addplot[mark={markers[i]}, {colors[i]}, densely dashed] coordinates "
        + "{\n"
    )
    output_latex(f, Gamma, L2_error_fixed_sigma_2[i])
    f.write("};\n\n")
    f.write(f"\\addlegendentry" + r"{$\phi$-FD2 $h=$" + f"{sizes[i]:.2f}" + "}\n")
f.write("\end{loglogaxis}\n")
f.write("\end{tikzpicture}")
f.write("\\quad")
f.write("\\begin{tikzpicture}")
f.write(
    "\\begin{loglogaxis}[name = ax1, width = .45\\textwidth, ylabel = Condition number,\n"
    + "xlabel = $\\gamma$,\n"
    + "legend style = {at={(1,1)},anchor=south east, legend columns =2,\n"
    + " /tikz/column 2/.style={column sep = 10pt}}]\n",
)

for i, h in enumerate(sizes):
    f.write(f"\\addplot[mark={markers[i]}, {colors[i]}] coordinates " + "{\n")
    output_latex(f, Gamma, Cond_fixed_sigma[i])
    f.write("};\n\n")
    f.write(f"\\addlegendentry" + r"{$\phi$-FD $h=$" + f"{sizes[i]:.2f}" + "}\n")
    f.write(
        f"\\addplot[mark={markers[i]}, {colors[i]}, densely dashed] coordinates "
        + "{\n"
    )
    output_latex(f, Gamma, Cond_fixed_sigma_2[i])
    f.write("};\n\n")
    f.write(f"\\addlegendentry" + r"{$\phi$-FD2 $h=$" + f"{sizes[i]:.2f}" + "}\n")
f.write("\end{loglogaxis}\n")
f.write("\end{tikzpicture}")

f.close()
