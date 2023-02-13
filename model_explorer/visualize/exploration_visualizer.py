import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# mainly taken from: https://pymoo.org/getting_started/part_4.html
class ExplorationVisualizer:

    def __init__(self, output_dir, res):
        self.output_dir = output_dir
        self.res = res

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.n_evals = []  # corresponding number of function evaluations\

        self.hist_cv = []  # constraint violation in each generation
        self.hist_cv_avg = [
        ]  # average constraint violation in the whole population

        self.hist_F = []  # the objective space values in each generation
        self.hist_X = []  # design variables of each generation

        self.hist_feasable_F = []  # all feasable values
        self.hist_feasable_X = []  # the thresholds for the feasable values

        self.hist_opt_F = []  # the optimum values for each generation
        self.hist_opt_X = (
            [])  # the thresholds for the optimum values for each generation

        self.global_opt_F = self.res.F  # the resulting pareto optima
        self.global_opt_X = self.res.X  # the thresholds for the pareto optima

        self._preprocess_hist(self.res)

    def _preprocess_hist(self, res):
        for algo in res.history:
            # store the number of function evaluations
            self.n_evals.append(algo.evaluator.n_eval)

            # # retrieve the optima from the algorithm
            self.hist_opt_X.append(algo.opt.get("X"))
            self.hist_opt_F.append(algo.opt.get("F"))

            # get the algorithms population
            pop = algo.pop

            # store the least contraint violation and the average in each population
            self.hist_cv.append(pop.get("CV").min())
            self.hist_cv_avg.append(pop.get("CV").mean())

            self.hist_X.append(algo.pop.get("X"))
            self.hist_F.append(pop.get("F"))

            # filter out only the feasible
            feas = np.where(pop.get("feasible"))[0]
            self.hist_feasable_X.append(pop.get("X")[feas])
            self.hist_feasable_X.append(pop.get("F")[feas])

        self.hist_F = np.abs(self.hist_F)

    def plot_layer_bit_resolution(self):
        plt.figure(figsize=(10, 5))

        layernames = self.res.problem.model.explorable_module_names

        for i, x in enumerate(self.global_opt_X):
            plt.plot(
                layernames,
                x,
                color="#808080",
                lw=0.5,
                label=f"F_1 objectives for limit acc: {self.global_opt_F[i][0]}",
            )

        plt.plot(
            layernames,
            np.mean(self.global_opt_X, axis=0),
            color="r",
            lw=0.7,
            label="Mean resolution",
            linestyle="--",
        )

        plt.xticks(rotation=90)
        plt.title("Layer dependent F_1")
        plt.xlabel("Layer name")
        plt.ylabel("F_1")
        # plt.legend() # produces too many values
        self._save("layers", plt)

    def plot_objective_space(self):
        plt.figure(figsize=(7, 5))
        plt.scatter(
            self.res.F[:, 0],
            self.res.F[:, 1],
            s=30,
            facecolors="none",
            edgecolors="blue",
        )
        plt.title("Objective Space")
        plt.xlabel("Accuracy (1-acc)")
        plt.ylabel("F_1 objective")
        self._save("objective_space", plt)

    def set_fontsize(self, fsize=14):
        matplotlib.rcParams.update({"font.size": fsize})

    def print_first_feasable_solution(self):
        k = np.where(np.array(self.hist_cv) <= 0.0)[0].min()
        print(
            f"At least one feasible solution in Generation {k} after {self.n_evals[k]} evaluations."
        )

    def plot_constraint_violation_over_n_gen(self):
        # replace this line by `hist_cv` if you like to analyze the least
        # feasible optimal solution and not the population
        vals = self.hist_cv_avg

        mins = np.where(np.array(vals) <= 0.0)[0]
        if len(mins) == 0:
            print(
                "Not enough feasible solutions for plotting constrain violations found."
            )
            return
        k = mins.min()
        print(
            f"Whole population feasible in Generation {k} after {self.n_evals[k]} evaluations."
        )

        plt.figure(figsize=(7, 5))
        plt.plot(self.n_evals,
                 vals,
                 color="black",
                 lw=0.7,
                 label="Avg. CV of Pop")
        plt.scatter(self.n_evals,
                    vals,
                    facecolor="none",
                    edgecolor="black",
                    marker="p")
        plt.axvline(self.n_evals[k],
                    color="red",
                    label="All Feasible",
                    linestyle="--")
        plt.title("Convergence")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Constraint Violation")
        plt.legend()
        self._save("cv_n_gen", plt)

    def plot_2d_pareto(self, acc_bound=0):
        fig, ax = plt.subplots(figsize=(10, 4))

        all_pts_flattened = self.hist_F.reshape(-1, self.hist_F.shape[-1])
        non_dom_pts_flattened = self.res.F.reshape(-1, self.hist_F.shape[-1])

        dominated_pts = [
            pt for pt in all_pts_flattened if pt not in non_dom_pts_flattened
        ]

        x_dominated_plot_list = [i[0] for i in dominated_pts]
        y_dominated_plot_list = [i[1] for i in dominated_pts]
        x_nondominated_plot_list = [i[0] for i in non_dom_pts_flattened]
        y_nondominated_plot_list = [i[1] for i in non_dom_pts_flattened]

        ax.scatter(
            x_dominated_plot_list,
            y_dominated_plot_list,
            marker=".",
            color="b",
            label="Dominated",
        )
        ax.scatter(
            x_nondominated_plot_list,
            y_nondominated_plot_list,
            marker="*",
            color="r",
            label="Non-Dominated",
        )
        ax.set_xlabel("Mean accuracy")
        ax.set_ylabel("F_1 objective")
        if acc_bound > 0:
            ax.set_xlim(left=acc_bound)
        ax.xaxis.set_tick_params(length=7)
        ax.yaxis.set_tick_params(length=7)
        plt.legend(loc="upper right")
        plt.subplots_adjust(bottom=0.20)
        if acc_bound > 0:
            self._save("2d_pareto_limit_{}".format(acc_bound), plt)
        else:
            self._save("2d_pareto", plt)
        plt.clf()

    def _save(self, name, figure, extensions=["svg", "png"]):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for extension in extensions:
            figure.savefig(
                os.path.join(self.output_dir, f"{name}.{extension}"),
                bbox_inches="tight",
            )
