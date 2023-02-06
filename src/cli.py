from pathlib import Path
import yaml
import click
from schema import SchemaError

from transfer_learning_criticality.schemas.config import config_schema
from transfer_learning_criticality.util.runner import run_experiment

@click.group()
@click.option("-c", "--configpath", default=Path(__file__).parent.parent / "config.yaml", type=click.Path(exists=True),  help="Path to the config file.")
@click.option("-t", "--train", default=False, help="Train model even if pretrained model exists.")
@click.pass_context
def cli(ctx, configpath, train):
    with open(configpath, "r") as configfile:
        config = yaml.safe_load(configfile)
        try:
            config_schema.validate(config)
        except SchemaError as error:
            print(f"Specified config is not valid. Validation error was {error}")
            exit(0)


        ctx.ensure_object(dict)
        ctx.obj["config"] = config
        ctx.obj["train"] = train


@cli.command()
@click.pass_context
@click.option("-p", "--root-path", "root_path", default=__file__, help="Root path for dataset and output files.")
def run_all(ctx, root_path):
    ctx.ensure_object(dict)
    experiments = ctx.obj["config"]["experiments"]

    for config in experiments:
        run_experiment(config, root_path, ~ctx.obj["train"])

@cli.command()
@click.argument("experiment_index", default=0)
@click.option("-p", "--root-path", "root_path", default=__file__, help="Root path for dataset and output files.")
@click.option("-s", "--show-summary", "show_model_summary", is_flag=True, default=False, help="Print model summary.")
@click.pass_context
def run(ctx, experiment_index, root_path, show_model_summary):
    
    ctx.ensure_object(dict)
    config = ctx.obj["config"]["experiments"][experiment_index]

    run_experiment(config, root_path, ~ctx.obj["train"], show_model_summary)
    

if __name__ == "__main__":
    cli()