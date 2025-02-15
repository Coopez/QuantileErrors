import optuna
import numpy as np

def objective(trial):
    steps = 100
    for step in  range(steps):
        x = trial.suggest_uniform('x', -10, 100) 
        value = x - step
        trial.report(value, step)
        if trial.should_prune():
            print(step)
            print(study.trials[-1].system_attrs)
            raise optuna.exceptions.TrialPruned()
    print(f"{trial.number}_{step}" )
    return value



# pruner = optuna.pruners.SuccessiveHalvingPruner(reduction_factor=2,min_early_stopping_rate=0)
pruner = optuna.pruners.HyperbandPruner(min_resource=1,max_resource=100,reduction_factor=3)
study = optuna.create_study(pruner = pruner)
study.optimize(objective, n_trials=100)
