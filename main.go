package main

import (
	"flag"
	"go/linear/gradient/learn"
	sliceLearn "go/linear/gradient/learn/slice"
	"go/linear/gradient/learn/vectorized"
	"go/linear/gradient/predict"
)

func main() {
	trainingFile := flag.String("training_file_path", "data_simple.txt", "a training csv file ")
	predictionFile := flag.String("prediction_file_path", "prediction_houses.txt", "a prediction csv file")
	resultFile := flag.String("result_file_path", "prediction_result.txt", "a result csv file")
	printCostFunction := flag.Bool("print_cost_function", false, "")
	alpha := flag.Float64("alpha_value", 1.2, "gradient step")
	iteration := flag.Int("iteration_number", 800, "training iteration")
	isVectorized := flag.Bool("vectorized_version", false, "")

	flag.Parse()

	config := learn.LearnConfiguration{
		PrintCostFunction: *printCostFunction,
		NumberIteration:   *iteration,
		Alpha:             *alpha,
		FileName:          *trainingFile,
	}
	var predictO predict.Predict
	var learnO learn.Learn
	if *isVectorized {

		learnO = vectorized.NewlearnVectorized()
	} else {
		learnO = sliceLearn.NewlearnSlice()
	}
	predictO, err := learnO.Learn(config)
	if err != nil {
		println(err)
		return
	}
	err = predictO.Predict(*predictionFile, *resultFile)
	if err != nil {
		println(err)
		return
	}

}
