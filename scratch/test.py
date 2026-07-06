import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt
from numpy._core.numeric import True_
from smac import RunHistory, Scenario
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)


class QuadraticFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (-5, 5), default=5)
        cs.add([x])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        x = config["x"]
        return x**2


def plot(runhistory: RunHistory, incumbent: Configuration) -> None:
    x = list(np.linspace(-5, 5, 100))
    y = [xi * xi for xi in x]
    plt.plot(x, y)

    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        x = config["x"]
        y = v.cost
        plt.scatter(x, y, c="blue", alpha=0.1, zorder=9999, marker="o")

    plt.scatter(
        incumbent["x"],
        incumbent["x"] * incumbent["x"],
        c="red",
        zorder=10000,
        marker="x",
    )
    plt.show()


if __name__ == "__main__":
    model = QuadraticFunction()
    scenario = Scenario(model.configspace, deterministic=True, n_trials=100)

    smac = HPOFacade(scenario, model.train, overwrite=True)
    incumbent = smac.optimize()
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    plot(smac.runhistory, incumbent)
