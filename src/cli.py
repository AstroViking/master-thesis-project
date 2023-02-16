from pathlib import Path
import yaml
import click

from transfer_learning_criticality.types.config import Config
from transfer_learning_criticality.util.runner import run_experiment

@click.group()
@click.option("-c", "--configpath", default=Path(__file__).parent.parent / "config.yaml", type=click.Path(exists=True),  help="Path to the config file.")
@click.pass_context
def cli(ctx, configpath):
    config = Config.from_yaml_file(configpath)
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.pass_context
@click.option("-p", "--root-path", "root_path", default=__file__, help="Root path for dataset and output files.")
def run_all(ctx, root_path):
    ctx.ensure_object(dict)
    config = ctx.obj["config"]

    for experiment in config.experiments:
        run_experiment(experiment, root_path)

@cli.command()
@click.argument("experiment_index", default=0)
@click.option("-p", "--root-path", "root_path", default=__file__, help="Root path for dataset and output files.")
@click.option("-s", "--show-summary", "show_model_summary", is_flag=True, default=False, help="Print model summary.")
@click.pass_context
def run(ctx, experiment_index, root_path, show_model_summary):
    
    ctx.ensure_object(dict)
    config = ctx.obj["config"]

    run_experiment(config.experiments[experiment_index], root_path, show_model_summary)
    

if __name__ == "__main__":
    cli()