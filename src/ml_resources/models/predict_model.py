import numpy as np

def calculate_accuracy(classifier, predictions, X_test, y_test, N=5):
  wrongPredictions=[]  
  correct_predictions=0
  correct_prediction_in_top_N_results=0
  for j, pred in enumerate(predictions):
     top_N_results_indices = np.argsort(pred)[-N:]
     top_N_results = classifier.classes_[top_N_results_indices]
     if top_N_results[-1] == y_test[j]:
         correct_predictions = correct_predictions + 1
     print([x for x in top_N_results])
     if y_test[j] in top_N_results:
         correct_prediction_in_top_N_results+=1
     else:
         wrongPredictions.append({"TopNPredictions": list(top_N_results), "True": list(y_test[j])[0]})
  print(f"Accuracy: {correct_predictions/len(predictions)}.")
  print(f"Correct or in top {N} results:{correct_prediction_in_top_N_results/len(predictions)}")
  return wrongPredictions