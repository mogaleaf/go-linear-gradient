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
	resultFile := flag.String("result_file_path", "prediction_result.txt", "a prediction csv file")
	printCostFunction := flag.Bool("print_cost_function", true, "")
	alpha := flag.Float64("alpha_value", 1.2, "")
	iteration := flag.Int("iteration_number", 800, "")
	readInteger := flag.Bool("use_integer", true, " if true read integer from data otherwise read float")

	theta, M, S, err := learn.Learn(*trainingFile, *readInteger, *alpha, *iteration, *printCostFunction)
	if err != nil {
		println(err.Error())
		return
	}

	resultMat, lines, err := predict.Predict(*predictionFile, *readInteger, theta, M, S)
	if err != nil {
		println(err.Error())
		return
	}

	err = writeData(resultMat, lines, *resultFile)
	if err != nil {
		println(err.Error())
		return
	}

}

func writeData(resultMat mat.Matrix, lines [][]string, resultFile string) error {
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
