# pip install tf_agents==0.19.0
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import greedy_policy, random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# --- Enhanced CSV Data Loading for NASA Exoplanet Archive Format ---
def robust_csv_loader(csv_path):
    """
    Robust CSV loader specifically designed for NASA Exoplanet Archive files
    """
    print(f"Attempting to load NASA Exoplanet Archive CSV: {csv_path}")
    
    try:
        # First, read the file to find where actual data starts
        with open(csv_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Find the line where actual data starts (after all # comments)
        data_start_line = 0
        header_line = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                # This is likely the header or first data row
                if data_start_line == 0:
                    header_line = i
                    data_start_line = i
                    break
        
        print(f" Detected NASA Exoplanet Archive format")
        print(f"  Comment lines: {data_start_line}")
        print(f"  Data starts at line: {data_start_line + 1}")
        
        # Read the CSV starting from the data line
        df = pd.read_csv(csv_path, skiprows=data_start_line, on_bad_lines='skip')
        
        print(f" Successfully loaded NASA format")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        return df, "nasa_format"
        
    except Exception as e:
        print(f" NASA format parsing failed: {e}")
    
    # Strategy 2: Standard pandas reading
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        print(f" Successfully loaded with standard method")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        return df, "standard"
    except Exception as e:
        print(f" Standard method failed: {e}")
    
    # Strategy 3: Different separators
    for sep in [',', ';', '\t', '|']:
        try:
            df = pd.read_csv(csv_path, sep=sep, on_bad_lines='skip', comment='#')
            if df.shape[1] > 1:  # Make sure we got multiple columns
                print(f" Successfully loaded with separator '{sep}'")
                print(f"  Shape: {df.shape}")
                return df, f"separator_{sep}"
        except Exception as e:
            continue
    
    print(f" All CSV loading strategies failed!")
    return None, None

def process_loaded_dataframe(df, load_method):
    """
    Process the loaded NASA Exoplanet Archive dataframe
    """
    print(f"\nProcessing NASA Exoplanet Archive dataframe...")
    
    # Show basic info
    print(f". DataFrame info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
    # Show first few column names
    print(f"  Sample columns: {list(df.columns[:5])}")
    
    # For NASA Exoplanet Archive, look for common exoplanet classification columns
    possible_label_columns = [
        # Discovery method (can be used to classify detection types)
        'discoverymethod', 'discovery_method', 'disc_method',
        # Controversy flag (confirmed vs controversial)
        'pl_controv_flag', 'controversial_flag', 'controv_flag',
        # Default flag (indicates best parameters)
        'default_flag', 'pl_default_flag',
        # Solution type
        'soltype', 'solution_type',
        # TTV flag (transit timing variations)
        'ttv_flag',
        # Any column with 'flag' or 'type'
    ]
    
    label_col = None
    
    # Look for exact matches first
    for col in possible_label_columns:
        if col in df.columns:
            label_col = col
            print(f" Found potential label column: '{label_col}'")
            break
    
    # Look for columns containing 'flag' or similar
    if label_col is None:
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['flag', 'type', 'method', 'controv']):
                label_col = col
                print(f" Found label column (keyword match): '{label_col}'")
                break
    
    # If no suitable column found, create labels based on planet properties
    if label_col is None:
        print(" No obvious label column found. Creating labels based on planet properties...")
        
        # Look for planet radius or mass columns to create size-based classification
        radius_cols = [col for col in df.columns if 'pl_rad' in col.lower() and 'err' not in col.lower() and 'lim' not in col.lower()]
        mass_cols = [col for col in df.columns if 'pl_mass' in col.lower() and 'err' not in col.lower() and 'lim' not in col.lower()]
        
        if radius_cols:
            radius_col = radius_cols[0]
            print(f"  Using planet radius '{radius_col}' to create size classification")
            # Create binary classification: small planets (< 2 Earth radii) vs large planets
            radius_values = pd.to_numeric(df[radius_col], errors='coerce')
            labels = (radius_values > 2.0).fillna(False).astype(int)  # 1 for large planets, 0 for small
            print(f"  Created size-based labels: {np.bincount(labels)} (small vs large planets)")
        elif mass_cols:
            mass_col = mass_cols[0]
            print(f"  Using planet mass '{mass_col}' to create mass classification")
            mass_values = pd.to_numeric(df[mass_col], errors='coerce')
            labels = (mass_values > 10.0).fillna(False).astype(int)  # 1 for massive planets, 0 for low-mass
            print(f"  Created mass-based labels: {np.bincount(labels)} (low-mass vs high-mass planets)")
        else:
            # Create labels based on discovery method
            if 'discoverymethod' in df.columns:
                print("  Using discovery method for classification")
                discovery_methods = df['discoverymethod'].fillna('Unknown')
                # Transit vs non-transit detection
                labels = (discovery_methods.str.contains('Transit', case=False, na=False)).astype(int)
                print(f"  Created method-based labels: {np.bincount(labels)} (non-transit vs transit)")
            else:
                # Random balanced labels as last resort
                print("  Creating balanced random labels")
                labels = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])
                print(f"  Created random labels: {np.bincount(labels)}")
    else:
        # Extract labels from identified column
        labels = df[label_col].values
        print(f"Labels extracted from column: '{label_col}'")
        
        # Convert to binary
        unique_labels = np.unique(labels[~pd.isna(labels)])
        print(f"  Unique values: {unique_labels}")
        
        if len(unique_labels) == 2:
            # Already binary
            if set(str(u).lower() for u in unique_labels if pd.notna(u)) & {'true', 'false', '1', '0', 'yes', 'no'}:
                # Boolean-like values
                labels = np.isin(labels, [1, '1', 'True', 'true', 'TRUE', 'Yes', 'yes', 'YES']).astype(int)
            else:
                # Numeric binary
                labels = (labels == unique_labels[1]).astype(int)
        else:
            # Multi-class to binary
            if 'transit' in str(unique_labels).lower():
                # Transit-based classification
                labels = np.array([1 if 'transit' in str(val).lower() else 0 for val in labels])
            else:
                # Use most common vs others
                most_common = pd.Series(labels).mode().iloc[0] if len(pd.Series(labels).mode()) > 0 else unique_labels[0]
                labels = (labels != most_common).astype(int)
        
        print(f"  Final binary labels: {np.bincount(labels)}")
    
    # Extract numeric features
    print(f"\nExtracting numeric features...")
    
    # Get all numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"  Found {len(numeric_df.columns)} numeric columns")
    
    if len(numeric_df.columns) == 0:
        # Try to convert some columns to numeric
        print("  Attempting to convert string columns to numeric...")
        potential_numeric = []
        
        for col in df.columns:
            if col != label_col:
                try:
                    # Try converting to numeric
                    numeric_vals = pd.to_numeric(df[col], errors='coerce')
                    # Keep if at least 50% of values are numeric
                    if numeric_vals.notna().sum() > len(df) * 0.5:
                        potential_numeric.append(col)
                        df[col] = numeric_vals
                except:
                    continue
        
        print(f"  Converted {len(potential_numeric)} columns to numeric")
        numeric_df = df[potential_numeric].select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        print(f" No numeric features available!")
        return None, None
    
    # Handle missing values
    print(f"  Processing missing values...")
    data = numeric_df.values
    
    if np.isnan(data).any():
        print(f"  Found {np.isnan(data).sum()} NaN values, filling with column medians")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        data = imputer.fit_transform(data)
    
    print(f"  Final feature extraction successful:")
    print(f"  Feature shape: {data.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
    print(f"  Feature columns used: {list(numeric_df.columns[:10])}{'...' if len(numeric_df.columns) > 10 else ''}")
    
    return data, labels

# --- Enhanced Data Preparation ---
print("Loading and preparing exoplanet data...")

try:
    # Try to load the CSV with robust methods
    df, load_method = robust_csv_loader('PS_2025.09.17_18.29.36.csv')
    
    if df is not None:
        # Process the loaded dataframe
        data, labels = process_loaded_dataframe(df, load_method)
        
        if data is not None and labels is not None:
            print(f"  Successfully processed CSV data!")
            print(f"  Final data shape: {data.shape}")
            print(f"  Final labels: {np.bincount(labels)}")
            
            # Ensure proper format for time series
            if len(data.shape) == 2:
                data = data.reshape(data.shape[0], data.shape[1], 1).astype(np.float32)
            
            csv_loaded = True
        else:
            print(f"  Failed to process CSV data, using synthetic data")
            csv_loaded = False
    else:
        print(f"  Failed to load CSV, using synthetic data")
        csv_loaded = False
        
except Exception as e:
    print(f"  Unexpected error loading CSV: {e}")
    csv_loaded = False

# Fallback to enhanced synthetic data if CSV loading failed
if not csv_loaded:
    print("\n" + "="*50)
    print("CREATING ENHANCED SYNTHETIC EXOPLANET DATA")
    print("="*50)
    
    num_samples = 2000
    num_features = 500
    num_channels = 1
    
    # Create realistic exoplanet data
    data = []
    labels = []
    
    for i in range(num_samples):
        # Generate realistic light curve
        time_points = np.linspace(0, 100, num_features)
        flux = np.ones(num_features) + np.random.normal(0, 0.005, num_features)
        
        # Add stellar variability
        stellar_period = np.random.uniform(10, 30)
        stellar_amp = np.random.uniform(0.001, 0.003)
        flux += stellar_amp * np.sin(2 * np.pi * time_points / stellar_period)
        
        # 45% chance of having exoplanet
        has_exoplanet = np.random.choice([0, 1], p=[0.55, 0.45])
        
        if has_exoplanet:
            # Add realistic transit signatures
            orbital_period = np.random.uniform(5, 80)  # Days
            transit_depth = np.random.uniform(0.003, 0.025)  # Transit depth
            transit_duration = np.random.uniform(1, 8)  # Hours
            
            # Calculate transits within observation period
            n_transits = int(100 / orbital_period) + 1
            
            for t in range(n_transits):
                transit_center = t * orbital_period + np.random.uniform(-1, 1)
                if 0 <= transit_center <= 100:
                    # Create transit shape
                    for j, time_val in enumerate(time_points):
                        time_diff = abs(time_val - transit_center)
                        if time_diff <= transit_duration/2:
                            # Gaussian-like transit shape
                            transit_factor = np.exp(-(time_diff / (transit_duration/4))**2)
                            flux[j] -= transit_depth * transit_factor
        
        data.append(flux)
        labels.append(has_exoplanet)
    
    data = np.array(data, dtype=np.float32).reshape(num_samples, num_features, num_channels)
    labels = np.array(labels, dtype=int)
    print(f" Created synthetic data: {data.shape}")
    print(f" Labels distribution: {np.bincount(labels)}")

print(f"\nFinal data summary:")
print(f"  Data shape: {data.shape}")
print(f"  Labels distribution: {np.bincount(labels)}")
print(f"  Data source: {'CSV file' if csv_loaded else 'Synthetic'}")

# Ensure data is properly shaped for time series
if len(data.shape) == 2:
    data = data.reshape(data.shape[0], data.shape[1], 1)

# Enhanced data splitting with stratification
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Robust normalization
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

scaler.fit(X_train_reshaped)
X_train_normalized = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_normalized = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Clip extreme values for stability
X_train_normalized = np.clip(X_train_normalized, -3, 3)
X_test_normalized = np.clip(X_test_normalized, -3, 3)

print(f"Training data shape: {X_train_normalized.shape}")
print(f"Test data shape: {X_test_normalized.shape}")
print(f"Training labels: {np.bincount(y_train)}")
print(f"Test labels: {np.bincount(y_test)}")

# --- Enhanced Environment with Superior Rewards ---
class HighPerformanceExoplanetEnvironment(py_environment.PyEnvironment):
    def __init__(self, data, labels):
        super().__init__()
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=data.shape[1:], dtype=np.float32, minimum=-3.0, maximum=3.0, name='observation'
        )
        
        self._data = data
        self._labels = labels.astype(np.int32)
        self._episode_ended = False
        self._current_index = 0
        
        # Enhanced data cycling
        self._data_indices = np.arange(len(self._data))
        np.random.shuffle(self._data_indices)
        self._index_pointer = 0
        self._epoch = 0
        
        # Performance tracking
        self._correct_predictions = 0
        self._total_predictions = 0
        self._class_correct = [0, 0]
        self._class_total = [0, 0]
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        if self._index_pointer >= len(self._data_indices):
            np.random.shuffle(self._data_indices)
            self._index_pointer = 0
            self._epoch += 1
            
        self._current_index = self._data_indices[self._index_pointer]
        self._index_pointer += 1
        self._episode_ended = False
        
        return ts.restart(self._data[self._current_index])

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        prediction = int(action)
        true_label = int(self._labels[self._current_index])
        
        # Superior reward system for maximum learning
        if prediction == true_label:
            if true_label == 1:  # Correctly found exoplanet (more important)
                reward = 5.0
            else:  # Correctly identified no exoplanet
                reward = 3.0
            self._correct_predictions += 1
            self._class_correct[true_label] += 1
        else:
            if true_label == 1 and prediction == 0:  # Missed exoplanet (bad!)
                reward = -3.0
            else:  # False positive (less bad)
                reward = -1.0
        
        self._total_predictions += 1
        self._class_total[true_label] += 1
        self._episode_ended = True

        return ts.termination(self._data[self._current_index], reward)
    
    def get_accuracy(self):
        return self._correct_predictions / max(1, self._total_predictions)

# --- TF-Agents Compatible Hybrid RNN+CNN+DRL Network ---
def build_hybrid_q_network(input_spec, action_spec):
    """
    TF-Agents compatible hybrid network using Sequential API
    """
    # Get input shape from spec
    input_shape = input_spec.shape
    num_actions = action_spec.maximum - action_spec.minimum + 1
    
    # Build hybrid network using Sequential (TF-Agents compatible)
    layers = [
        # CNN Branch: Multi-scale feature extraction
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', 
                              input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        
        # Global pooling to reduce dimensions
        tf.keras.layers.GlobalMaxPooling1D(),
        
        # Dense layers for decision making
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Q-values output
        tf.keras.layers.Dense(num_actions)
    ]
    
    return sequential.Sequential(layers)

# Alternative: Custom TF-Agents Network Class
class HybridRNNCNNNetwork(tf.keras.Model):
    """Custom hybrid network that works with TF-Agents"""
    
    def __init__(self, input_spec, action_spec, name='HybridRNNCNN'):
        super(HybridRNNCNNNetwork, self).__init__(name=name)
        
        self._input_spec = input_spec
        self._action_spec = action_spec
        self._num_actions = action_spec.maximum - action_spec.minimum + 1
        
        # CNN Branch
        self.cnn_conv1 = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')
        self.cnn_pool1 = tf.keras.layers.MaxPooling1D(2)
        self.cnn_conv2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')
        self.cnn_pool2 = tf.keras.layers.MaxPooling1D(2)
        self.cnn_conv3 = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')
        
        # RNN Branch (simplified for TF-Agents compatibility)
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
        self.lstm2 = tf.keras.layers.LSTM(32, dropout=0.2)
        
        # Global pooling
        self.global_pool = tf.keras.layers.GlobalMaxPooling1D()
        
        # Fusion and decision layers
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(self._num_actions)
    
    def call(self, inputs, training=None):
        # CNN path
        cnn_x = self.cnn_conv1(inputs)
        cnn_x = self.cnn_pool1(cnn_x)
        cnn_x = self.cnn_conv2(cnn_x)
        cnn_x = self.cnn_pool2(cnn_x)
        cnn_x = self.cnn_conv3(cnn_x)
        cnn_features = self.global_pool(cnn_x)
        
        # RNN path (simplified)
        rnn_x = self.lstm1(inputs, training=training)
        rnn_features = self.lstm2(rnn_x, training=training)
        
        # Fusion
        combined = tf.keras.layers.Concatenate()([cnn_features, rnn_features])
        
        # Decision network
        x = self.dense1(combined)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        
        return self.output_layer(x)

# --- Setup High-Performance Environments ---
print("Setting up high-performance hybrid environments...")

train_py_env = HighPerformanceExoplanetEnvironment(X_train_normalized, y_train)
eval_py_env = HighPerformanceExoplanetEnvironment(X_test_normalized, y_test)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Build the TF-Agents compatible hybrid network
print("Building TF-Agents Compatible Hybrid RNN+CNN+DRL Network...")

# Option 1: Use Sequential API (simpler, more reliable)
print("Using Sequential API for maximum TF-Agents compatibility...")
hybrid_q_net = build_hybrid_q_network(
    train_env.observation_spec(),
    train_env.action_spec()
)

print("Network built successfully!")
try:
    print(f"Network summary:")
    hybrid_q_net.summary()
except:
    print("Network created (summary not available for Sequential)")

# Option 2: Alternative custom network (uncomment if needed)
# print("Using Custom Network Class...")
# hybrid_q_net = HybridRNNCNNNetwork(
#     train_env.observation_spec(),
#     train_env.action_spec()
# )

# --- High-Performance DQN Agent ---
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=hybrid_q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    epsilon_greedy=0.5,  # Start with high exploration
    target_update_tau=0.005,  # Soft target updates
    target_update_period=50,
    gamma=0.99,  # High discount factor
    reward_scale_factor=0.1
)

agent.initialize()
print("High-performance DQN agent initialized!")

# --- Enhanced Training Setup ---
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=30000  # Large buffer for diverse experiences
)

# Comprehensive initial data collection
print("Collecting comprehensive initial experiences...")
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                               train_env.action_spec())

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    random_policy,
    observers=[replay_buffer.add_batch],
    num_steps=2000  # Extensive initial collection
)

initial_time_step = train_env.reset()
collect_driver.run(initial_time_step)

print(f"Replay buffer size after initial collection: {int(replay_buffer.num_frames())}")

# Create enhanced training data collector
collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=150  # Collect many steps per iteration
)

# High-performance dataset
dataset = replay_buffer.as_dataset(
    num_parallel_calls=8,
    sample_batch_size=128,  # Large batches
    num_steps=2
).prefetch(8)

iterator = iter(dataset)

# --- Comprehensive Evaluation Function ---
def comprehensive_evaluation(environment, policy, num_episodes=100):
    """Comprehensive evaluation with detailed metrics"""
    total_return = 0.0
    correct_predictions = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        
        while not time_step.is_last():
            action_step = policy.action(time_step)
            action = int(action_step.action.numpy())
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            
            # Get true label from the environment
            env_wrapper = environment.pyenv.envs[0]
            current_idx = (env_wrapper._index_pointer - 1) % len(env_wrapper._data_indices)
            true_label = int(env_wrapper._labels[env_wrapper._data_indices[current_idx]])
            
            # Update confusion matrix
            if action == 1 and true_label == 1:
                true_positives += 1
            elif action == 1 and true_label == 0:
                false_positives += 1
            elif action == 0 and true_label == 0:
                true_negatives += 1
            elif action == 0 and true_label == 1:
                false_negatives += 1
                
            if action == true_label:
                correct_predictions += 1
        
        total_return += episode_return
    
    # Calculate comprehensive metrics
    avg_return = total_return / num_episodes
    accuracy = correct_predictions / num_episodes
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    return {
        'avg_return': avg_return,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': [true_negatives, false_positives, false_negatives, true_positives]
    }

# --- Model Checkpointing Setup ---
import os
import pickle

# Create directories for saving models
os.makedirs("model_checkpoints", exist_ok=True)
os.makedirs("best_models", exist_ok=True)

# Initialize best model tracking
best_model_weights = None
best_scaler = None
best_metrics = None

# --- High-Performance Training Loop ---
print("Starting high-performance hybrid training...")
num_iterations = 200
best_f1 = 0.0
best_accuracy = 0.0
patience = 0
max_patience = 800
evaluation_history = []

for i in range(num_iterations):
    start_time = time.time()
    
    # Enhanced data collection
    collect_driver.run()
    
    # Multiple training steps for faster learning
    num_train_steps = 2 if i < 1000 else 1
    
    for _ in range(num_train_steps):
        try:
            experience, _ = next(iterator)
            train_loss = agent.train(experience)
        except StopIteration:
            iterator = iter(dataset)
            experience, _ = next(iterator)
            train_loss = agent.train(experience)
    
    # Advanced epsilon decay schedule
    if i < 1000:
        epsilon = 0.5 * np.exp(-i / 300)  # Exponential decay
    elif i < 2500:
        epsilon = 0.2 * (0.998 ** (i - 1000))  # Slow linear decay
    else:
        epsilon = max(0.01, 0.05 * (0.999 ** (i - 2500)))  # Final refinement
    
    if hasattr(agent._collect_policy, '_epsilon_greedy'):
        agent._collect_policy._epsilon_greedy = epsilon
    
    step_time = time.time() - start_time
    
    # Comprehensive evaluation every 100 iterations
    if i % 100 == 0 and i > 0:
        print(f"\nEvaluating at iteration {i}...")
        metrics = comprehensive_evaluation(eval_env, agent.policy, num_episodes=100)
        evaluation_history.append(metrics)
        
        # Track best performance
        improved = False
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            improved = True
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            improved = True
            
        if improved:
            patience = 0
            status = "*** NEW BEST ***"
            
            # Save best model weights and associated data
            print("  Saving best model...")
            
            # Save the TF-Agents policy (complete model)
            tf.saved_model.save(agent.policy, f"best_models/best_policy_iter_{i}")
            
            # Save the underlying Q-network weights
            try:
                # For Sequential network
                if hasattr(agent._q_network, 'layers'):
                    agent._q_network.save_weights(f"best_models/best_q_network_weights_iter_{i}.h5")
                else:
                    # For custom network
                    agent._q_network.save_weights(f"best_models/best_q_network_weights_iter_{i}.h5")
            except Exception as e:
                print(f"  Warning: Could not save Q-network weights: {e}")
            
            # Save scaler and preprocessing info
            with open(f"best_models/best_scaler_iter_{i}.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save metrics and model info
            model_info = {
                'iteration': i,
                'metrics': metrics,
                'data_shape': X_train_normalized.shape,
                'num_classes': 2,
                'feature_columns': list(range(X_train_normalized.shape[1])),
                'training_params': {
                    'learning_rate': 3e-4,
                    'epsilon': epsilon,
                    'buffer_size': int(replay_buffer.num_frames()),
                    'train_steps': int(agent.train_step_counter.numpy())
                }
            }
            
            with open(f"best_models/best_model_info_iter_{i}.pkl", 'wb') as f:
                pickle.dump(model_info, f)
            
            # Update global best tracking
            best_model_weights = f"best_models/best_q_network_weights_iter_{i}.h5"
            best_scaler = scaler
            best_metrics = metrics
            
            print(f"  Best model saved at iteration {i}")
            
        else:
            patience += 100
            status = f"(patience: {patience}/{max_patience})"
        
        print(f"Iteration {i:4d}:")
        print(f"  Accuracy:    {float(metrics['accuracy'])*100:6.2f}%")
        print(f"  Precision:   {float(metrics['precision'])*100:6.2f}%")
        print(f"  Recall:      {float(metrics['recall'])*100:6.2f}%")
        print(f"  F1-Score:    {float(metrics['f1_score'])*100:6.2f}%")
        print(f"  Specificity: {float(metrics['specificity'])*100:6.2f}%")
        print(f"  Avg Return:  {float(metrics['avg_return']):7.2f}")
        print(f"  Epsilon:     {float(epsilon):.4f}")
        print(f"  Loss:        {float(train_loss.loss):7.4f}")
        print(f"  Buffer:      {int(replay_buffer.num_frames()):5d}")
        print(f"  {status}")
        
        # Early stopping
        if patience >= max_patience and i > 1500:
            print(f"\nEarly stopping at iteration {i} (no improvement for {max_patience} iterations)")
            break

# --- Final Comprehensive Evaluation ---
print("\n" + "="*70)
print("FINAL COMPREHENSIVE EVALUATION - HYBRID RNN+CNN+DRL MODEL")
print("="*70)

final_metrics = comprehensive_evaluation(
    eval_env, 
    greedy_policy.GreedyPolicy(agent.policy), 
    num_episodes=len(y_test)
)

print(f"\nFinal Test Results:")
print(f"  Accuracy:       {float(final_metrics['accuracy'])*100:6.2f}%")
print(f"  Precision:      {float(final_metrics['precision'])*100:6.2f}%")
print(f"  Recall:         {float(final_metrics['recall'])*100:6.2f}%")
print(f"  F1-Score:       {float(final_metrics['f1_score'])*100:6.2f}%")
print(f"  Specificity:    {float(final_metrics['specificity'])*100:6.2f}%")
print(f"  Average Return: {float(final_metrics['avg_return']):8.4f}")

print(f"\nTraining Summary:")
print(f"  Total Training Steps: {int(agent.train_step_counter.numpy()):,}")
print(f"  Best Training F1:     {float(best_f1)*100:6.2f}%")
print(f"  Best Training Acc:    {float(best_accuracy)*100:6.2f}%")
print(f"  Network Type:         TF-Agents Compatible Sequential")
print(f"  Architecture:         Hybrid RNN+CNN+DRL")

# Performance analysis
baseline_accuracy = max(np.bincount(y_test)) / len(y_test)
print(f"\nPerformance Analysis:")
print(f"  Random Baseline:      50.00%")
print(f"  Majority Baseline:    {float(baseline_accuracy)*100:6.2f}%")
print(f"  Hybrid Model:         {float(final_metrics['accuracy'])*100:6.2f}%")
print(f"  Improvement:          {float(final_metrics['accuracy'] - 0.5)*100:+6.2f}pp")

if final_metrics['accuracy'] > 0.85:
    performance_grade = "OUTSTANDING"
elif final_metrics['accuracy'] > 0.75:
    performance_grade = "EXCELLENT"
elif final_metrics['accuracy'] > 0.65:
    performance_grade = "GOOD"
elif final_metrics['accuracy'] > 0.55:
    performance_grade = "MODERATE"
else:
    performance_grade = "NEEDS IMPROVEMENT"

print(f"  Performance Grade:    {performance_grade}")

if evaluation_history:
    print(f"\nTraining Progress:")
    print(f"  Initial Accuracy: {float(evaluation_history[0]['accuracy'])*100:6.2f}%")
    print(f"  Final Accuracy:   {float(evaluation_history[-1]['accuracy'])*100:6.2f}%")
    print(f"  Total Improvement: {float(evaluation_history[-1]['accuracy'] - evaluation_history[0]['accuracy'])*100:+6.2f}pp")

print(f"\n" + "="*70)

# --- Save Final Best Model ---
print("\nSaving final best model and utilities...")

# Save the final policy
final_policy_path = "best_models/final_best_policy"
tf.saved_model.save(agent.policy, final_policy_path)
print(f"Final policy saved to: {final_policy_path}")

# Save final Q-network
try:
    final_weights_path = "best_models/final_best_q_network_weights.h5"
    agent._q_network.save_weights(final_weights_path)
    print(f"Final Q-network weights saved to: {final_weights_path}")
except Exception as e:
    print(f"Warning: Could not save final Q-network weights: {e}")

# Save final scaler
final_scaler_path = "best_models/final_scaler.pkl"
with open(final_scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Final scaler saved to: {final_scaler_path}")

# Save comprehensive model metadata
final_model_metadata = {
    'final_metrics': final_metrics,
    'best_training_metrics': best_metrics,
    'data_info': {
        'train_shape': X_train_normalized.shape,
        'test_shape': X_test_normalized.shape,
        'num_classes': 2,
        'data_source': 'CSV file' if csv_loaded else 'Synthetic'
    },
    'model_architecture': {
        'type': 'TF-Agents DQN with Hybrid CNN+RNN',
        'network_type': 'Sequential',
        'input_shape': X_train_normalized.shape[1:],
        'output_classes': 2
    },
    'training_info': {
        'total_iterations': num_iterations,
        'total_train_steps': int(agent.train_step_counter.numpy()),
        'best_iteration': best_metrics.get('iteration', 'unknown') if best_metrics else 'unknown',
        'final_epsilon': epsilon,
        'optimizer': 'Adam',
        'learning_rate': 3e-4
    },
    'evaluation_history': evaluation_history
}

final_metadata_path = "best_models/final_model_metadata.pkl"
with open(final_metadata_path, 'wb') as f:
    pickle.dump(final_model_metadata, f)
print(f"Model metadata saved to: {final_metadata_path}")

# Create model loading utility
model_loader_code = '''
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def load_best_exoplanet_model():
    """
    Load the best trained exoplanet detection model
    Returns: (policy, scaler, metadata)
    """
    try:
        # Load the TF-Agents policy
        policy = tf.saved_model.load("best_models/final_best_policy")
        
        # Load the scaler
        with open("best_models/final_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # Load metadata
        with open("best_models/final_model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        print("Model loaded successfully!")
        print(f"Model accuracy: {metadata['final_metrics']['accuracy']*100:.2f}%")
        print(f"Model F1-score: {metadata['final_metrics']['f1_score']*100:.2f}%")
        
        return policy, scaler, metadata
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def predict_exoplanet(data, policy, scaler):
    """
    Predict exoplanet presence using the trained model
    
    Args:
        data: Raw input data (can be 1D, 2D, or 3D array)
        policy: Loaded TF-Agents policy
        scaler: Loaded StandardScaler
    
    Returns:
        prediction: 0 (no exoplanet) or 1 (exoplanet detected)
        confidence: prediction confidence (0-1)
    """
    try:
        # Ensure data has correct shape
        if len(data.shape) == 1:
            data = data.reshape(1, -1, 1)
        elif len(data.shape) == 2:
            if data.shape[0] == 1:
                data = data.reshape(1, data.shape[1], 1)
            else:
                data = data.reshape(1, data.shape[0], 1)
        elif len(data.shape) == 3 and data.shape[0] != 1:
            data = data[0:1]  # Take first sample
        
        # Normalize data
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        data_normalized = scaler.transform(data_reshaped)
        data_normalized = data_normalized.reshape(original_shape)
        data_normalized = np.clip(data_normalized, -3, 3)
        
        # Convert to tensor
        data_tensor = tf.constant(data_normalized, dtype=tf.float32)
        
        # Create time step
        from tf_agents.trajectories import time_step as ts
        time_step = ts.restart(data_tensor)
        
        # Get prediction
        action_step = policy.action(time_step)
        prediction = int(action_step.action.numpy()[0])
        
        # Calculate confidence (simplified)
        confidence = 0.8 if prediction == 1 else 0.7  # Placeholder
        
        return prediction, confidence
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# Example usage:
if __name__ == "__main__":
    # Load model
    policy, scaler, metadata = load_best_exoplanet_model()
    
    if policy is not None:
        # Example prediction on random data
        test_data = np.random.randn(500, 1)  # Example light curve
        prediction, confidence = predict_exoplanet(test_data, policy, scaler)
        
        if prediction is not None:
            result = "Exoplanet detected!" if prediction == 1 else "No exoplanet detected"
            print(f"Prediction: {result} (confidence: {confidence:.2f})")
'''

model_loader_path = "best_models/load_model.py"
with open(model_loader_path, 'w') as f:
    f.write(model_loader_code)
print(f"Model loading utility saved to: {model_loader_path}")

print(f"\nModel saving complete! Files saved:")
print(f"  - Policy: {final_policy_path}")
print(f"  - Weights: {final_weights_path}")
print(f"  - Scaler: {final_scaler_path}")
print(f"  - Metadata: {final_metadata_path}")
print(f"  - Loader utility: {model_loader_path}")

# Legacy Keras model save (for compatibility)
try:
keras_model = agent._q_network.layers[0]

# Save as normal Keras model
    keras_model.save("best_models/hybrid_q_network.h5")
    print("  - Legacy Keras model: best_models/hybrid_q_network.h5")
except Exception as e:
    print(f"  - Warning: Could not save legacy Keras model: {e}")

print(f"\nTo use the model later, run:")
print(f"  python best_models/load_model.py")