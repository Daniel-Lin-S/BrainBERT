"""Train BrainBERT with Hydra; add +local=machine.local for managed fits.

Managed local configuration supplies storage roots and explicit scientific
settings. Legacy group-only invocations retain the original task interface.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name=None)
def main(cfg: DictConfig) -> None:
    """Dispatch a resolved configuration to the shared training workflow."""
    if "storage" in cfg:
        from util.experiments.execution import execute

        execute(cfg)
        return
    import tasks
    from runner import Runner

    task = tasks.setup_task(cfg.task)
    task.load_datasets(cfg.data, cfg.preprocessor)
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    runner = Runner(cfg.exp.runner, task, model, criterion)
    try:
        best_model = runner.train()
        runner.test(best_model)
    finally:
        runner.progress.close()
        if runner.logger is not None:
            runner.logger.close()


if __name__ == "__main__":
    main()
