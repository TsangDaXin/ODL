import json

file_path = 'c:/Users/60122/OneDrive/Desktop/ODL_ASSIGNMENT/ODL/notebooks/04_comparison (3).ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Check if targets exist
if 'def evaluate_probs(probs, y_true, class_names, model_name):' not in text:
    print("Function definition not found.")
else:
    text = text.replace(
        'def evaluate_probs(probs, y_true, class_names, model_name):\\n',
        'def evaluate_probs(probs, y_true, class_names, model_name, precomputed_loss=None):\\n'
    )
    
    text = text.replace(
        'loss = float(keras.losses.CategoricalCrossentropy()(y_true_onehot, probs).numpy())\\n',
        'if precomputed_loss is not None:\\n        loss = precomputed_loss\\n    else:\\n        loss = float(keras.losses.CategoricalCrossentropy()(y_true_onehot, probs).numpy())\\n'
    )
    
    text = text.replace(
        'result_cnn = evaluate_probs(cnn_test_probs, y_test_true, CLASS_NAMES, \\"Custom CNN C3 (Tuned)\\")\\n',
        'cnn_loss, _ = cnn_c3.evaluate(test_ds, verbose=0)\\n    result_cnn = evaluate_probs(cnn_test_probs, y_test_true, CLASS_NAMES, \\"Custom CNN C3 (Tuned)\\", precomputed_loss=cnn_loss)\\n'
    )
    
    text = text.replace(
        'result_resnet = evaluate_probs(resnet_test_probs, y_test_true, CLASS_NAMES, \\"ResNet50 C2 (Improved)\\")\\n',
        'resnet_loss, _ = resnet_c2.evaluate(test_ds, verbose=0)\\n    result_resnet = evaluate_probs(resnet_test_probs, y_test_true, CLASS_NAMES, \\"ResNet50 C2 (Improved)\\", precomputed_loss=resnet_loss)\\n'
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Notebook updated successfully.")
