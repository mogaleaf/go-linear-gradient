package cost

import (
	"gonum.org/v1/gonum/mat"
)

func ComputeCostvectorized(X mat.Matrix, y mat.Matrix, theta mat.Matrix) (float64, error) {
	r, _ := X.Dims()
	m, _ := y.Dims()
	_, c := theta.Dims()

	//Calculate Theta*X our hypothesis
	XTheta := mat.NewDense(r, c, nil)
	XTheta.Mul(X, theta)

	//Remove the training result to get h(x)-y
	XThetaMinusY := mat.NewDense(r, c, nil)
	XThetaMinusY.Sub(XTheta, y)

	//(h(x)-y)x)^2
	XThetaMinusYTransposeProduct := mat.NewDense(c, c, nil)
	XThetaMinusYTransposeProduct.Mul(XThetaMinusY.T(), XThetaMinusY)
	//1/2m(h(x)-y)x)^2
	a := 1 / float64(2*m)

	//Should be 1*1 dim
	return a * XThetaMinusYTransposeProduct.At(0, 0), nil
}
