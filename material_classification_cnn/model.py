from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(16, 9, activation='relu', padding='valid', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Conv1D(32, 9, activation='relu', padding='valid', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(pool_size=16),
        Dropout(0.1),

        Conv1D(64, 3, activation='relu', padding='valid', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='valid', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        Dropout(0.1),

        Conv1D(128, 3, activation='relu', padding='valid', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
