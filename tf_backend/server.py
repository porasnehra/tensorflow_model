import os
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from models.tf_model import create_tf_model

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # grab the run settings
    total_rounds = context.run_config.get("num-server-rounds", 10)
    client_fraction = context.run_config.get("fraction-train", 1.0)
    
    # setup the blank global model to start the whole process
    print("--> [Server] Initializing the master neural net...")
    model = create_tf_model()
    initial_weights = ArrayRecord(model.get_weights())
    
    # using standard federated averaging
    strategy = FedAvg(fraction_train=client_fraction)
    

    
    # start the actual training loop
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_weights,
        num_rounds=total_rounds,
    )
    
    # training is done, let's save the final brain to disk
    os.makedirs("results", exist_ok=True)
    save_path = "results/final_model.keras"
    
    final_weights = result.arrays.to_numpy_ndarrays()
    model.set_weights(final_weights)
    model.save(save_path)
    
