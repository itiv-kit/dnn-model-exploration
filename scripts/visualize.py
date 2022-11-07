



def render_results(pickle_file=None, res_obj=None):
    # WARNING: this is not backwards compatible
    res = res_obj

    if res == None:
        # try to load it from file
        f = open(pickle_file, "rb")
        res = pickle.load(f)

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    exp_vis = ExplorationVisualizer(
        RESULTS_DIR + f"/visuals/{date_str}/",
        str(res.problem.quantization_model._model.__class__.__name__),
        res,
        res.problem.quantization_model.activation_fake_quantizers.keys(),
    )

    exp_vis.print_first_feasable_solution()
    exp_vis.plot_constraint_violation_over_n_gen()
    exp_vis.plot_2d_pareto()
    exp_vis.plot_layer_bit_resolution()
    exp_vis.plot_objective_space()