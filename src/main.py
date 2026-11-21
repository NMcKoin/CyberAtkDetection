import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
  #set seed for reproducable results
  seed = 42

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Read in and evaluate data
  df = pd.read_csv('./data/unsw-cyber.csv')

  # display basic dataset info
  print(df.head())
  print(df.shape)
  print(df.isnull().sum())
  # 1500 missing values in attack_cat but its a non numeric column

  # Separate features and target
  y = df["label"]
  X = df.drop("label", axis=1)

  # Remove non numeric columns
  X = X.select_dtypes(include=[np.number])
  print("Number of features:", X.shape[1])

  # Split data into training, validation, and test sets
  # Ration is 60% train, 20% validation, 20% test
  X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=seed,
    stratify=y
  )

  X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=seed,
    stratify=y_temp
  )

  # Save original test data for inference
  X_test_original = X_test.copy()

  
  # Standar Scaler - fit only on training data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)

  # Convert to tensors
  X_train = torch.FloatTensor(X_train)
  X_val = torch.FloatTensor(X_val)
  X_test = torch.FloatTensor(X_test)

  #conver labels into tensors for training
  y_train = torch.FloatTensor(
    y_train.values.reshape(-1, 1)
  )

  y_val = torch.FloatTensor(
    y_val.values.reshape(-1, 1)
  )

  y_test = torch.FloatTensor(
    y_test.values.reshape(-1, 1)
  )


  # MLP Class
  class MLPClassifier(nn.Module):

    def __init__(self, input_size, hidden_layers, dropout_rate):
        super().__init__()

        layers = []

        previous_size = input_size

        for size in hidden_layers:

            layers.append(
                nn.Linear(previous_size, size)
            )

            layers.append(nn.ReLU())

            layers.append(
                nn.Dropout(dropout_rate)
            )

            previous_size = size

        layers.append(
            nn.Linear(previous_size, 1)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
       return self.network(x)

  # Hyperparameter config
  configs = [

    {
        "hidden_layers": [64],
        "learning_rate": 0.001,
        "dropout": 0.2
    },

    {
        "hidden_layers": [128],
        "learning_rate": 0.001,
        "dropout": 0.3
    },

    {
        "hidden_layers": [128, 64],
        "learning_rate": 0.001,
        "dropout": 0.3
    },

    {
        "hidden_layers": [256, 128],
        "learning_rate": 0.0005,
        "dropout": 0.4
    }
  ]

  # Result storage
  results = []
  best_accuracy = 0
  best_model = None
  best_config = None

  # Train and evaluate a model for each configuration
  for config in configs:
    model = MLPClassifier(
      input_size=X_train.shape[1],
      hidden_layers=config["hidden_layers"],
      dropout_rate=config["dropout"]
    )

    # Binary classification loss function
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer using the current learning rate
    optimizer = torch.optim.Adam(
      model.parameters(),
      lr=config["learning_rate"]
    )

    # Train the model for 100 epochs
    epochs = 100

    for epoch in range(epochs):

      # set model to training mode
      model.train()
      #Clear gradients from last iteration
      optimizer.zero_grad()
      # Generate predictions for training samples
      outputs = model(X_train)

      # Computer training loss
      train_loss = criterion(
        outputs,
        y_train
      )

      # Update network weights
      train_loss.backward()
      optimizer.step()

    # Evaluate performance on validation data
    model.eval()

    with torch.no_grad():

      val_outputs = model(X_val)
      val_loss = criterion(
        val_outputs,
        y_val
      )


    # Accuaracy
    # Convert probabilities into binary predictions
    predictions = (
    torch.sigmoid(val_outputs) >= 0.5
    ).float()

    # Calculate validation accuaracy
    accuracy = accuracy_score(
    y_val.numpy(),
    predictions.numpy()
    )


    # Save metrics from experiment
    results.append({
      "hidden_layers":
        str(config["hidden_layers"]),
      "learning_rate":
        config["learning_rate"],
      "dropout":
        config["dropout"],
      "training_loss":
        train_loss.item(),
      "validation_loss":
        val_loss.item(),
      "accuracy":
        accuracy
    })

    # Keep the best performing configuration
    if accuracy > best_accuracy:

      best_accuracy = accuracy
      best_config = config
      best_model = model

  # Save the best model
  torch.save(
    best_model.state_dict(),
    "./output/best_model.pth"
  )

  print("Best Configuration:")
  print(best_config)

  results_df = pd.DataFrame(results)

  results_df.to_csv(
    "./output/sweep_results.csv",
    index=False
  )

  print(results_df)

# Eval the best model on unseen test data
  best_model.eval()

  with torch.no_grad():

    test_outputs = best_model(X_test)

    test_loss = criterion(
        test_outputs,
        y_test
      )

    test_predictions = (
        torch.sigmoid(test_outputs) >= 0.5
      ).float()

    test_accuracy = accuracy_score(
        y_test.numpy(),
        test_predictions.numpy()
      )

  print(f"Test Loss: {test_loss.item():.4f}")

  print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
  )

  # Reusable inference pipeline for new net traffic samples
  def predict_sample(
    sample,
    model,
    scaler
):
    # Apply the same scaling transformation from training
    sample_scaled = scaler.transform(
        [sample]
    )

    #convert the sample into a tensor
    sample_tensor = torch.FloatTensor(
        sample_scaled
    )

    # Run inference without computing gradients
    model.eval()

    with torch.no_grad():

        output = model(sample_tensor)

        probability = torch.sigmoid(
            output
        )

        prediction = (
            1
            if probability.item() >= 0.5
            else 0
        )

    return prediction

  # Classify 3 unseen samples from test set
  print("\nInference Results:")

  for i in range(3):

    prediction = predict_sample(
        X_test_original.iloc[i].values,
        best_model,
        scaler
    )

    print(
        f"Sample {i+1}: "
        f"{prediction}"
    )


main()

# Which hyperparameter had the largest effect on performance?
  # The hidden layer architecture had the largest effect on performance.
  # The deeper networks ([128, 64] and [256, 128]) achieved the highest
  # validation accuracy and lowest validation loss. Changes to learning
  # rate and dropout had a smaller impact compared to increasing the
  # network depth and width.

# Did any configurations appear to overfit?
  # No configuration showed significant overfitting. The training and
  # validation losses remained relatively close, and validation accuracy
  # stayed consistently high across all experiments. This suggests the
  # models generalized well to the validation data.

# Did your model perform similarly on the validation set and test set?
  # What does this suggest about the model's ability to generalize to unseen data?
  # Yes. The best model achieved a validation accuracy of approximately
  # 99.67% and a test accuracy of 99.33%. Since the values are very
  # similar, this suggests the model generalized well and maintained
  # strong performance on unseen data.

# Why must the same scaler from training be reused during prediction?
  # The model was trained using standardized feature values created by the
  # StandardScaler. Reusing the same scaler ensures that new samples are
  # transformed in the same way as the training data. Using a different
  # scaler or unscaled data would change the feature distributions and may
  # result in inaccurate predictions.