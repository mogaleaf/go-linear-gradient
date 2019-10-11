package predict

// Predict an output from a trained Learning..
type Predict interface {
	Predict(inputFilename string, outputFilename string) error
}
