import tensorflow as tf

def create_tf_model(input_shape=(36,), learning_rate=0.001):
    # This is our custom brain for catching mule accounts!
    # It takes 36 distinct pieces of data (our engineered features)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Layer 1: Look for initial hidden patterns
        tf.keras.layers.Dense(32, activation='relu'),
        
        # We drop 20% of the neurons randomly to prevent the model from memorizing the data
        tf.keras.layers.Dropout(0.2),
        
        # Layer 2: Condense the patterns
        tf.keras.layers.Dense(16, activation='relu'),
        
        # Final Output: Give us a clean percentage from 0% (Safe) to 100% (Fraud!)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Teach the model how to learn from its mistakes
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def load_tf_model(model_path: str):
    # Just a quick helper function to reload a saved brain
    return tf.keras.models.load_model(model_path)
