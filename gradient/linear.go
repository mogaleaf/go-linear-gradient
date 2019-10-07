package gradient

import "gonum.org/v1/gonum/mat"

func LinearGradient(X mat.Matrix, y mat.Matrix, theta mat.Matrix, alpha float64, num_iters int) (mat.Matrix, error) {
	for i := 0; i < num_iters; i++ {

		r, cx := X.Dims()
		m, _ := y.Dims()
		_, c := theta.Dims()
		H := mat.NewDense(c, r, nil)
		H.Mul(theta.T(), X.T())
		DIFF := mat.NewDense(r, c, nil)
		DIFF.Sub(H.T(), y)
		SUM := mat.NewDense(c, cx, nil)
		SUM.Mul(DIFF.T(), X)

		RESULT := mat.NewDense(cx, c, nil)
		RESULT.Scale(alpha*(1/float64(m)), SUM.T())
		RESULT.Sub(theta, RESULT)

		theta = RESULT

	}
	return theta, nil
}
