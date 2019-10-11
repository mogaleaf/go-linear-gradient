package learn

import "go/linear/gradient/predict"

// Learn a training file with alpha and number of iteration. Return object to predict an output.
type Learn interface {
	Learn(LearnConfiguration) (predict.Predict, error)
}

type LearnConfiguration struct {
	TrainingFileName   string
	PredictionFileName string
	Alpha              float64
	NumberIteration    int
	PrintCostFunction  bool
}
