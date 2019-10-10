package hypothesis

import "gonum.org/v1/gonum/mat"

func ComputeHypothesis(x []float64, theta []float64) float64 {
	result := theta[0]
	for i := 1; i < len(theta); i++ {
		result += theta[i] * x[i-1]
	}
	return result
}

func ComputeHypothesisVectorized(theta mat.Matrix, data mat.Matrix) (mat.Matrix, error) {
	r, _ := data.Dims()
	_, c := theta.Dims()
	dense := mat.NewDense(c, r, nil)
	dense.Mul(theta.T(), data.T())
	return dense, nil
}
