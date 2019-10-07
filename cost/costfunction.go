package cost

import (
	"gonum.org/v1/gonum/mat"
)

func ComputeCost(X mat.Matrix, y mat.Matrix, theta mat.Matrix) (float64, error) {
	r, _ := X.Dims()
	m, _ := y.Dims()
	_, c := theta.Dims()
	XTheta := mat.NewDense(r, c, nil)
	XTheta.Mul(X, theta)

	XThetaMinusY := mat.NewDense(r, c, nil)
	XThetaMinusY.Sub(XTheta, y)

	XThetaMinusYTransposeProduct := mat.NewDense(c, c, nil)
	XThetaMinusYTransposeProduct.Mul(XThetaMinusY.T(), XThetaMinusY)
	a := float64((1 / float64(2*m)))

	return a * XThetaMinusYTransposeProduct.At(0, 0), nil
}
