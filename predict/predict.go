package predict

// Predict an output from a trained Learning..
type Predict interface {
	Predict(resultData chan float64)
	PredictLength() int
}
