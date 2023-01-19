from template import Template
import plots

def main():
    model = Template(
        "test_plot",
        {"image_size": (28, 28)},
        log_intevall=1
    )
    model.gather_statistics()
    
    plots.accuracy_over_time(model.training_stats[:,0], model.training_stats[:,1], 
                             model.training_stats[:,3], color="blue")
    plots.plt_save(model.out_path, f"{plots.accuracy_over_time.__name__}.png")
    plots.loss_over_time(model.training_stats[:,0], model.training_stats[:,1], 
                             model.training_stats[:,2], color="red")
    plots.plt_save(model.out_path, f"{plots.loss_over_time.__name__}.png")


if __name__ == "__main__":
    main()