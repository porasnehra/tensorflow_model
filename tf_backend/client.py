import tensorflow as tf
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from data.data_loader import prepare_federated_data
from models.tf_model import create_tf_model

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # clear out old sessions so memory doesn't blow up
    tf.keras.backend.clear_session()
    
    # figure out which bank branch we are pretending to be right now
    branch_id = context.node_config.get("partition-id", 0)
    print(f"--> [Client] Booting up training for Bank Branch #{branch_id}")
    
    # grab the data for this specific branch
    all_client_data, _, _, _, _ = prepare_federated_data()
    x_train = all_client_data[branch_id]['X_train']
    y_train = all_client_data[branch_id]['y_train']
    
    # setup the neural net and slap the latest global weights on it
    model = create_tf_model()
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
    
    # grab hyperparameters from the server config (fallback to sensible defaults)
    epochs = context.run_config.get("local-epochs", 3)
    batch_size = context.run_config.get("batch-size", 32)
    
    print(f"--> [Client] Training on {len(x_train)} accounts for {epochs} epochs...")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # pluck out the final metrics to send back
    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]
    
    print(f"--> [Client] Done! Final Accuracy: {final_acc:.2%}")
    
    # bundle everything up to ship back to the central server
    model_record = ArrayRecord(model.get_weights())
    metrics = {
        "num-examples": len(x_train),
        "train_loss": float(final_loss),
        "train_acc": float(final_acc)
    }
    
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
