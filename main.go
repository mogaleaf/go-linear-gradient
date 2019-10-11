package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"go/linear/gradient/learn"
	sliceLearn "go/linear/gradient/learn/slice"
	"go/linear/gradient/learn/vectorized"
	"go/linear/gradient/predict"
	"os"
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
		PrintCostFunction:  *printCostFunction,
		NumberIteration:    *iteration,
		Alpha:              *alpha,
		TrainingFileName:   *trainingFile,
		PredictionFileName: *predictionFile,
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
	length := predictO.PredictLength()
	float64s := make(chan float64, length)
	go predictO.Predict(float64s)

	writeData(*resultFile, float64s)

}

func writeData(resultFile string, data chan float64) error {
	f, err := os.Create(resultFile)
	if err != nil {
		return err
	}
	defer f.Close()
	writer := csv.NewWriter(f)

	for i := range data {
		writer.Write([]string{fmt.Sprintf("%0.10f", i)})
		println(fmt.Sprintf("prediction %0.10f", i))

	}
	writer.Flush()
	return nil
}
