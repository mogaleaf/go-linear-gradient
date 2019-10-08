package predict

import (
	"encoding/csv"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func Predict(predictionFile string, readInteger bool, theta mat.Matrix, M mat.Matrix, S mat.Matrix) (mat.Matrix, [][]string, error) {
	f, err := os.Open(predictionFile)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, err
	}

	predictData := mat.NewDense(len(lines), len(lines[0])+1, nil)
	for i, line := range lines {
		for j, data := range line {
			var f float64
			if readInteger {
				rI, err := strconv.ParseInt(data, 10, 64)
				if err != nil {
					return nil, nil, err
				}
				f = float64(rI)
			} else {
				f, err = strconv.ParseFloat(data, 64)
				if err != nil {
					return nil, nil, err
				}
			}
			if err != nil {
				return nil, nil, err
			}
			if j == 0 {
				predictData.Set(i, 0, 1)
			}
			predictData.Set(i, j+1, f)

		}
	}

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

	result, err := calcPredict(theta, predictData)
	if err != nil {
		return nil, nil, err
	}
	return result, lines, err
}

func calcPredict(theta mat.Matrix, data mat.Matrix) (mat.Matrix, error) {
	r, _ := data.Dims()
	_, c := theta.Dims()
	dense := mat.NewDense(c, r, nil)
	dense.Mul(theta.T(), data.T())
	return dense, nil
}
