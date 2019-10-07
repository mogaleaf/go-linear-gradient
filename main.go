package main

import (
	"encoding/csv"
	"fmt"
	"go/linear/gradient/gradient"
	"go/linear/gradient/normalize"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func learn(fileName string) (mat.Matrix, mat.Matrix, mat.Matrix, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		panic(err)
	}

	dataLoaded := make([][]int64, len(lines))
	X := mat.NewDense(len(lines), len(lines[0]), nil)
	y := mat.NewDense(len(lines), 1, nil)
	theta := mat.NewDense(len(lines[0]), 1, nil)
	// Loop through lines & turn into object
	for i, line := range lines {
		dataLoaded[i] = make([]int64, len(line))
		for j, data := range line {
			//TODO parse float
			f, err := strconv.ParseInt(data, 10, 64)
			if err != nil {
				return nil, nil, nil, err
			}
			if j < len(line)-1 {
				if j == 0 {
					X.Set(i, 0, 1)
				}
				X.Set(i, j+1, float64(f))
			} else {
				y.Set(i, 0, float64(f))
			}

		}

	}
	XNorm, M, S, err := normalize.Normalize(X)

	matrix, err := gradient.LinearGradient(XNorm, y, theta, 1.2, 100000)
	if err != nil {
		return nil, nil, nil, err
	}
	return matrix, M, S, nil
}

func predict(theta mat.Matrix, data mat.Matrix) (float64, error) {
	dense := mat.NewDense(1, 1, nil)
	dense.Mul(theta.T(), data.T())
	return dense.At(0, 0), nil
}

func main() {
	theta, M, S, err := learn("data.txt")
	if err != nil {
		println(err.Error())
		return
	}
	data := []float64{1, 1650, 3}
	predictData := mat.NewDense(1, 3, data)
	predictData.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return v
		}
		return v - M.At(0, j-1)
	}, predictData)
	predictData.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return v
		}
		return v / S.At(0, j-1)
	}, predictData)
	f, err := predict(theta, predictData)
	println(fmt.Sprintf("prediction for 1650 3 = %0.10f", f))

}
