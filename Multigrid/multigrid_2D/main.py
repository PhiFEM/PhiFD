import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from run_solvers import Solver_multigrid
from ref_agent import RefAgent
import os
import seaborn as sns
from matplotlib.patches import Ellipse
import scipy.sparse.linalg as sla

sns.set_theme()
sns.set_context("paper")

pp = print
plt_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
cmap = sns.color_palette()


def confidence_ellipse(x, y, ax, facecolor="none", **kwargs):

    if x.size != y.size:
        raise ValueError("incompatible sizes")

    ell_radius_x = np.std(x, axis=1)
    ell_radius_y = np.std(y, axis=1)
    centers_x = np.mean(x, axis=1)
    centers_y = np.mean(y, axis=1)
    for i in range(len(centers_x)):
        ellipse = Ellipse(
            (centers_x[i], centers_y[i]),
            width=ell_radius_x[i] * 2,
            height=ell_radius_y[i] * 2,
            facecolor=facecolor,
            **kwargs,
        )
        ax.add_patch(ellipse)


fast_test = False


def go_creation():
    force_new_data = False

    N_final = 2200
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

    test_cases = ["circle_cos"]  # ["circle_cos", "ellipsis_cos", "ellipsis_cos_exp"]
    for test_case in test_cases:

        print(f"Test case : {test_case}")
        if not os.path.exists(f"./results_{test_case}/list_L2_direct_{test_case}.npy"):
            # force_new_data = True

            if not os.path.exists(f"./results_{test_case}"):
                os.makedirs(f"./results_{test_case}")
            agents = {
                N: RefAgent(N, N_final=N_final, test_case=test_case, iterative=False)
                for N in Ns
            }
            L2 = [agents[N].result["L2"] for N in Ns]
            y = [agents[N].result["duration"] for N in Ns]
            Loo = [agents[N].result["Loo"] for N in Ns]

            np.save(f"./results_{test_case}/list_L2_direct_{test_case}.npy", L2)
            np.save(f"./results_{test_case}/list_times_direct_{test_case}.npy", y)
            np.save(f"./results_{test_case}/list_Loo_direct_{test_case}.npy", Loo)

        if not os.path.exists(
            f"./results_{test_case}/list_L2_iterative_{test_case}.npy"
        ):
            # force_new_data = True

            if not os.path.exists(f"./results_{test_case}"):
                os.makedirs(f"./results_{test_case}")
            agents = {
                N: RefAgent(N, N_final=N_final, test_case=test_case, iterative=True)
                for N in Ns
            }
            L2 = [agents[N].result["L2"] for N in Ns]
            y = [agents[N].result["duration"] for N in Ns]
            Loo = [agents[N].result["Loo"] for N in Ns]

            np.save(f"./results_{test_case}/list_L2_iterative_{test_case}.npy", L2)
            np.save(f"./results_{test_case}/list_times_iterative_{test_case}.npy", y)
            np.save(f"./results_{test_case}/list_Loo_iterative_{test_case}.npy", Loo)

        L2_direct = np.load(f"./results_{test_case}/list_L2_direct_{test_case}.npy")
        Loo_direct = np.load(f"./results_{test_case}/list_Loo_direct_{test_case}.npy")
        times_direct = np.load(
            f"./results_{test_case}/list_times_direct_{test_case}.npy"
        )
        L2_iterative = np.load(
            f"./results_{test_case}/list_L2_iterative_{test_case}.npy"
        )
        Loo_iterative = np.load(
            f"./results_{test_case}/list_Loo_iterative_{test_case}.npy"
        )
        times_iterative = np.load(
            f"./results_{test_case}/list_times_iterative_{test_case}.npy"
        )

        for explicit_expression in [False]:
            list_xs_L2, list_xs_Loo, list_ys = [], [], []
            index = 0
            for N in N_multigrid:
                print(f"N = {N}")
                index += 1
                if explicit_expression:
                    if (
                        not os.path.exists(
                            f"./grid_search_results_explicit_expression/grid_search_output_{test_case}_{N}.csv"
                        )
                    ) or force_new_data:
                        if not os.path.exists(
                            "./grid_search_results_explicit_expression"
                        ):
                            os.makedirs("./grid_search_results_explicit_expression")
                        solver = Solver_multigrid(
                            N,
                            N_final=N_final,
                            test_case=test_case,
                            explicit_expression=explicit_expression,
                        )
                        results = solver.solve()
                        print(results)
                        results_df = pd.DataFrame.from_dict([results])
                        results_df.to_csv(
                            f"./grid_search_results_explicit_expression/grid_search_output_{test_case}_{N}.csv"
                        )
                else:
                    if (
                        not os.path.exists(
                            f"./grid_search_results_interpolate_phi/grid_search_output_{test_case}_{N}.csv"
                        )
                    ) or force_new_data:
                        if not os.path.exists("./grid_search_results_interpolate_phi"):
                            os.makedirs("./grid_search_results_interpolate_phi")
                        solver = Solver_multigrid(
                            N,
                            N_final=N_final,
                            test_case=test_case,
                            explicit_expression=explicit_expression,
                        )
                        results = solver.solve()
                        print(results)
                        results_df = pd.DataFrame.from_dict([results])
                        results_df.to_csv(
                            f"./grid_search_results_interpolate_phi/grid_search_output_{test_case}_{N}.csv"
                        )
            if explicit_expression:
                for N in N_multigrid:
                    results_df = pd.read_csv(
                        f"./grid_search_results_explicit_expression/grid_search_output_{test_case}_{N}.csv",
                        index_col=0,
                    )
                    print(results_df)
                    x_L2, x_Loo, time = get_point(results_df)
                    list_ys.append(time)
                    list_xs_L2.append(x_L2)
                    list_xs_Loo.append(x_Loo)

                np.save(
                    f"./results_{test_case}/list_L2_multigrid_explicit_expression_{test_case}.npy",
                    list_xs_L2,
                )
                np.save(
                    f"./results_{test_case}/list_Loo_multigrid_explicit_expression_{test_case}.npy",
                    list_xs_Loo,
                )
                np.save(
                    f"./results_{test_case}/list_times_multigrid_explicit_expression_{test_case}.npy",
                    list_ys,
                )
            else:
                for N in N_multigrid:
                    results_df = pd.read_csv(
                        f"./grid_search_results_interpolate_phi/grid_search_output_{test_case}_{N}.csv",
                        index_col=0,
                    )
                    print(results_df)
                    x_L2, x_Loo, time = get_point(results_df)
                    list_ys.append(time)
                    list_xs_L2.append(x_L2)
                    list_xs_Loo.append(x_Loo)

                np.save(
                    f"./results_{test_case}/list_L2_multigrid_interpolate_phi_{test_case}.npy",
                    list_xs_L2,
                )
                np.save(
                    f"./results_{test_case}/list_Loo_multigrid_interpolate_phi_{test_case}.npy",
                    list_xs_Loo,
                )
                np.save(
                    f"./results_{test_case}/list_times_multigrid_interpolate_phi_{test_case}.npy",
                    list_ys,
                )
        fig, (ax0) = plt.subplots(1, 1, figsize=(10, 10))
        # ax0.set_title("Time VS $L_2$ error", fontsize=20)
        for ax in [ax0]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$L^2$ relative error", fontsize=16)
            ax.set_ylabel("Computation time (s)", fontsize=16)

        ax0.plot(
            L2_direct, times_direct, "-o", markersize=4, label=f"Direct", c=cmap[0]
        )  #  "o-", c="gray",
        ax0.plot(
            L2_iterative,
            times_iterative,
            "-+",
            markersize=8,
            label=f"Iterative",
            c=cmap[1],
        )  # "*--", c="k",
        if os.path.exists(
            f"./results_{test_case}/list_times_multigrid_interpolate_phi_{test_case}.npy"
        ):
            L2_error = np.load(
                f"./results_{test_case}/list_L2_multigrid_interpolate_phi_{test_case}.npy"
            )
            Loo_error = np.load(
                f"./results_{test_case}/list_Loo_multigrid_interpolate_phi_{test_case}.npy"
            )
            times = np.load(
                f"./results_{test_case}/list_times_multigrid_interpolate_phi_{test_case}.npy"
            )
            ax0.plot(
                L2_error,
                times,
                "-d",
                markersize=8,
                c=cmap[2],
                # "x--",
                # color="r",
                label=r"Multigrid",
            )

        # if os.path.exists(
        #     f"./results_{test_case}/list_times_multigrid_explicit_expression_{test_case}.npy"
        # ):
        #     L2_error = np.load(
        #         f"./results_{test_case}/list_L2_multigrid_explicit_expression_{test_case}.npy"
        #     )
        #     Loo_error = np.load(
        #         f"./results_{test_case}/list_Loo_multigrid_explicit_expression_{test_case}.npy"
        #     )
        #     times = np.load(
        #         f"./results_{test_case}/list_times_multigrid_explicit_expression_{test_case}.npy"
        #     )
        #     ax0.plot(
        #         L2_error,
        #         times,
        #         "-x",
        #         markersize=8,
        #         # "x--",
        #         # color="r",
        #         label=r"Multigrid explicit expressions",
        #     )
        for i, label in enumerate(N_multigrid):
            ax0.text(
                L2_error[i],
                times[i] - 0.5,
                f"$N_0$={label}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=14,
            )
        for i, label in enumerate(Ns):
            ax0.text(
                L2_direct[i],
                times_direct[i],
                f"$N_0$={label}",
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize=14,
            )
            ax0.text(
                L2_iterative[i],
                times_iterative[i],
                f"$N_0$={label}",
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize=14,
            )
        for ax in [ax0]:
            ax.legend(ncol=1, fontsize=16)
            ax.grid(axis="both", which="both")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"output_{test_case}.pdf")
        # show_fig = len(test_cases) == 1
        # if show_fig:
            # plt.show()


def get_point(df):
    if len(df) != 1:
        raise Exception("Ah ca c'est dommage")

    line = df.loc[0]
    eL2 = line["L2"]
    time = line["duration"]
    eLoo = line["Loo"]
    return eL2, eLoo, time


if __name__ == "__main__":
    go_creation()
