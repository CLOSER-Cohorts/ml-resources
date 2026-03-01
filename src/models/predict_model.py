def calculateAccuracy(classifier, predictions, X_test, y_test, N):
  wrongPredictions=[]  
  correct_predictions=0
  correct_prediction_in_top_N_results=0
  for j in range(len(predictions)):
     res = sorted(range(len(predictions[j])), key = lambda sub: predictions[j][sub])[-N:]
     if classifier.classes_[res][-1]==y_test[j]:
         correct_predictions = correct_predictions + 1
     print([x for x in classifier.classes_[res]])
     if y_test[j] in [x for x in classifier.classes_[res]]:
         correct_prediction_in_top_N_results+=1
     else:
         wrongPredictions.append({"TopNPredictions": list(classifier.classes_[res]), "True": list(y_test[j])[0]})
  print(f"Accuracy: {correct_predictions/len(predictions)}.")
  print(f"Correct or in top {N} results:{correct_prediction_in_top_N_results/len(predictions)}")
  return wrongPredictions

predictions=trainedModel.predict(list(dataForTrainingAndTest['X_test']['QuestionEmbedding'].values))
predictions_with_probabilities=trainedModel.predict_proba(list(dataForTrainingAndTest['X_test']['QuestionEmbedding'].values))
calculateAccuracy(trainedModel, 
    predictions_with_probabilities, 
    dataForTrainingAndTest['X_test']['QuestionEmbedding'].values, 
    dataForTrainingAndTest['y_test'].values,
    N)