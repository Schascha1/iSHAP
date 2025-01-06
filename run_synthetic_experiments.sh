iterations=100
for sampling in "uniform" "normal"
    do
    for model in "mult" "sin"
        do
        echo "Iteration,F1-GREEDY,F1-GREEDY2,Pair-Precision-GREEDY,Pair-Precision-GREEDY2,Runtime-GREEDY,Runtime-GREEDY2,Steps-GREEDY,Steps-GREEDY2,Vsteps-GREEDY,Vsteps-GREEDY2,Pair-Recall-GREEDY,Pair-Recall-GREEDY2" >  results/$sampling-$model.csv
        echo "Iteration,F1-FULL,F1-FULL2,F1-NSHAP,Pair-Precision-FULL,Pair-Precision-FULL2,Pair-Precision-NSHAP,Runtime-FULL,Runtime-FULL2,Runtime-NSHAP,Steps-FULL,Steps-FULL2,Steps-NSHAP,Vsteps-FULL,Vsteps-FULL2,Vsteps-NSHAP,Pair-Recall-FULL,Pair-Recall-FULL2,Pair-Recall-NSHAP" >  results/$sampling-$model-nshap.csv
        python3 synthetic_experiments.py $sampling $model $iterations &
        done
    done