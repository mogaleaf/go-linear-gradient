package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"go/linear/gradient/learn"
	"go/linear/gradient/predict"
	"os"

	"gonum.org/v1/gonum/mat"
)

func main() {
	trainingFile := flag.String("training_file_path", "data.txt", "a training csv file ")
	predictionFile := flag.String("prediction_file_path", "prediction.txt", "a prediction csv file")
	resultFile := flag.String("result_file_path", "prediction_result.txt", "a result csv file")
	printCostFunction := flag.Bool("print_cost_function", false, "")
	alpha := flag.Float64("alpha_value", 1.2, "gradient step")
	iteration := flag.Int("iteration_number", 800, "training iteration")
	vectorized := flag.Bool("vectorized_version", false, "")

	flag.Parse()

	if *vectorized {
		theta, M, S, err := learn.LearnVectorized(*trainingFile, *alpha, *iteration, *printCostFunction)
		if err != nil {
			println(err.Error())
			return
		}

		resultMat, lines, err := predict.PredictVectorized(*predictionFile, theta, M, S)
		if err != nil {
			println(err.Error())
			return
		}

		err = writeDataVectorized(resultMat, lines, *resultFile)
		if err != nil {
			println(err.Error())
			return
		}
	} else {

		theta, M, S, err := learn.Learn(*trainingFile, *alpha, *iteration, true)

		resultMat, _, err := predict.Predict(*predictionFile, theta, M, S)
		if err != nil {
			println(err.Error())
			return
		}

		err = writeData(resultMat, *resultFile)
		if err != nil {
			println(err.Error())
			return
		}

	}

}

func writeDataVectorized(resultMat mat.Matrix, lines [][]string, resultFile string) error {
	f, err := os.Create(resultFile)
	if err != nil {
		return err
	}
	defer f.Close()
	writer := csv.NewWriter(f)
	r, c := resultMat.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			newLine := append(lines[j], fmt.Sprintf("%0.10f", resultMat.At(i, j)))
			writer.Write(newLine)
			println(fmt.Sprintf("prediction %s = %f", lines[j], resultMat.At(i, j)))
		}

	}
	writer.Flush()
	return nil
}

func writeData(result []float64, resultFile string) error {
	f, err := os.Create(resultFile)
	if err != nil {
		return err
	}
	defer f.Close()
	writer := csv.NewWriter(f)
	for i := 0; i < len(result); i++ {
		writer.Write([]string{fmt.Sprintf("%0.10f", result[i])})
		println(fmt.Sprintf("prediction %0.10f", result[i]))

	}
	writer.Flush()
	return nil
}
