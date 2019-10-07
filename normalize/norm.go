package normalize

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Normalize(X mat.Matrix) (mat.Matrix, mat.Matrix, mat.Matrix, error) {
	r, c := X.Dims()
	N := mat.NewDense(r, c, nil)
	S := mat.NewDense(1, c-1, nil)
	M := mat.NewDense(1, c-1, nil)

	for j := 1; j < c; j++ {
		var max float64
		min := math.MaxFloat64
		var sum float64
		for i := 0; i < r; i++ {
			if max < X.At(i, j) {
				max = X.At(i, j)
			}
			if min > X.At(i, j) {
				min = X.At(i, j)
			}
			sum += X.At(i, j)
		}
		mean := float64(sum / float64(r))
		M.Set(0, j-1, mean)
		s := max - min

		S.Set(0, j-1, s)

		for i := 0; i < r; i++ {

			N.Set(i, j, (X.At(i, j)-mean)/s)
		}
	}
	for i := 0; i < r; i++ {
		N.Set(i, 0, 1)
	}
	return N, M, S, nil
}
