# 3 Write a program to develop a single-layer Perceptron network in Python using weights [0.2, 0.4, 0.2] and bias -0.5 to predict a student's movie-going decision based on three binary input features and evaluate its prediction accuracy. CO1
import itertools

# Weights and Bias as per prompt
weights = [0.2, 0.4, 0.2]
bias = -0.5

# Step function (Activation)
def step_function(z):
    if z >= 0:
        return 1
    else:
        return 0

# Perceptron calculation
def predict(inputs):
    z = sum(i * w for i, w in zip(inputs, weights)) + bias
    return step_function(z)

# Ground Truth: Suppose the student ONLY goes if 'Heroine' is present AND one other factor
# This is what we compare our model against to find real accuracy
def actual_decision(inputs):
    hero, heroine, climate = inputs
    return 1 if (heroine == 1 and (hero == 1 or climate == 1)) else 0

# Test all combinations
combinations = list(itertools.product([0, 1], repeat=3))
correct_predictions = 0

print("Inputs (H, Hn, C) | Predicted | Actual")
for x in combinations:
    pred = predict(x)
    actual = actual_decision(x)

    if pred == actual:
        correct_predictions += 1

    print(f"{x}           | {pred}         | {actual}")

accuracy = (correct_predictions / len(combinations)) * 100
print(f"\nModel Accuracy: {accuracy}%")
