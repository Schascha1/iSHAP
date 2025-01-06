tests=100
points=2
for model in "linear" "knn" "svm" "mlp" "rf"
    do
    echo "Dataset, iSHAP, SHAP, NSHAP-Part, NSHAP-Full" >  results/accuracy/$model.csv
    for dataset in "breast_cancer" "california" "diabetes" "credit"  "insurance" "student" "life"
        do
        python3 experiments.py $dataset $model $tests $points &
        done
    wait
    echo "Dataset,LIME" >  results/accuracy/lime_$model.csv
    for dataset in "breast_cancer" "california" "diabetes" "credit"  "insurance" "student" "life"
        do
        python3 lime_experiment.py $dataset $model $tests $points &
        done
done