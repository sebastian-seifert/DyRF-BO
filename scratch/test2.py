import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from ConfigSpace import Configuration, ConfigurationSpace, Integer
from smac import Scenario, RunHistory
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade

# 1. Load data
data = load_iris()
X, y = data.data, data.target

# 2. Objective function
def train_random_forest(config: Configuration, seed: int = 0) -> float:
    rf = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=seed,
        n_jobs=-1
    )
    # SMAC MINIMIZES, so we return 1 - accuracy
    scores = cross_val_score(rf, X, y, cv=5)
    return 1 - np.mean(scores)

def plot_search_history(runhistory: RunHistory, incumbent: Configuration):
    """
    Visualizes the search points. 
    X-axis: n_estimators
    Y-axis: max_depth
    Color: Cost (Darker/Blue = Better accuracy)
    """
    n_estimators_list = []
    max_depth_list = []
    costs = []

    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        n_estimators_list.append(config["n_estimators"])
        max_depth_list.append(config["max_depth"])
        costs.append(v.cost)

    plt.figure(figsize=(10, 6))
    
    # Scatter plot of all trials
    sc = plt.scatter(n_estimators_list, max_depth_list, c=costs, cmap="viridis_r", s=100, edgecolors='black', alpha=0.7)
    plt.colorbar(sc, label='Cost (1 - Accuracy)')
    
    # Mark the best configuration (incumbent) with a red star
    plt.scatter(incumbent["n_estimators"], incumbent["max_depth"], 
                c="red", marker="*", s=300, label="Best Found", edgecolors='black')

    plt.title("SMAC Search History: Random Forest HPO")
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    # 3. Define the 2D Search Space
    cs = ConfigurationSpace(seed=42)
    n_estimators = Integer("n_estimators", (10, 200), default=100)
    max_depth = Integer("max_depth", (1, 30), default=5)
    cs.add([n_estimators, max_depth])

    # 4. Setup Scenario (40 trials to explore 2D space)
    scenario = Scenario(cs, n_trials=40, deterministic=True)

    # 5. Optimize
    smac = HPOFacade(scenario, train_random_forest, overwrite=True)
    incumbent = smac.optimize()

    # 6. Final Results
    print("\n--- Optimization Results ---")
    print(f"Best n_estimators: {incumbent['n_estimators']}")
    print(f"Best max_depth:    {incumbent['max_depth']}")
    
    best_accuracy = 1 - smac.validate(incumbent)
    print(f"Final Accuracy:    {best_accuracy:.4f}\n")

    # 7. Visualize
    plot_search_history(smac.runhistory, incumbent)
