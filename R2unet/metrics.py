import numpy as np
import torch


def evaluate(ground_truth, predictions):

    #flatten arrays to 1D
    ground_truth = torch.flatten(ground_truth).numpy()
    predictions = torch.flatten(predictions).numpy()
    
    
    Acc, SE, SP, F1, Dice =   get_eval_metrics(ground_truth, predictions)
    return Acc, SE, SP, F1, Dice
    
    
def get_eval_metrics(ground_truth, predictions):
    classes = np.unique(ground_truth)
    for i in (classes):
        #calculate TP, FP and FN for each class
        wrongs = ground_truth != predictions
        TP = np.sum(ground_truth[ground_truth == predictions] == i)
        FN = np.sum(ground_truth[wrongs] == i)
        FP = np.sum(predictions[wrongs] == i)
        TN = np.sum(ground_truth[ground_truth == predictions] != i)
        
        if (TP + FP + FN == 0):
            dice_coefficient = 0
            F1_score = 0
        elif (TP+FP+FN !=0):
            dice_coefficient = (2*TP)/((2*TP) + FP + FN)
            F1_score = TP/(TP + 0.5*(FP + FN))
            
        if (TP + FN == 0):
            sensitivity = 0
        elif (TP+FN!=0):
            sensitivity = TP/(TP + FN)
            
        if (TN + FP == 0):
            specificity = 0
        elif (TN+FP!=0):
            specificity = TN/(TN + FP)
        
        
        
        if i == 0:
            positives = TP + TN
            negatives = FP + FN
            
            dice_coefficients = np.array(dice_coefficient)
            SE = np.array(sensitivity)
            SP = np.array(specificity)
            F1 = np.array(F1_score)
            
        elif(i!=0):
            positives = positives + TP + TN
            negatives = negatives + FP + FN
            
            dice_coefficients = np.append(dice_coefficients, dice_coefficient)
            SE = np.append(SE, sensitivity)
            SP = np.append(SP, specificity)
            F1 = np.append(F1, F1_score)
           
        
    Acc = positives/(positives + negatives)
    
    Dice = (np.sum(dice_coefficients))/np.sum(classes)
    SE = (np.sum(SE))/np.sum(classes)
    SP = (np.sum(SP))/np.sum(classes)
    F1 = (np.sum(F1))/np.sum(classes)
    return Acc, SE, SP, F1, Dice