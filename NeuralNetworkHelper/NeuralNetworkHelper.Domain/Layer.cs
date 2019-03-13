using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkHelper.NeuralNetworkHelper.Domain
{
    public enum OutputType
    {
        OutputLayer,
        HiddenLayer
    }

    public class Layer
    {
        // perceptrons should be created already and come with weights, eta, and bias already assigned
        public Layer(OutputType outputType, List<Perceptron<List<double>>> perceptrons)
        {
            Perceptrons = perceptrons;
            PerceptronWeights = new List<double>();
            LittleError = new List<double>();
            Outputs = new List<double>();

            OutputLayerFlag = outputType;
        }

        public List<Perceptron<List<double>>> Perceptrons { get; set; }
        public List<double> PerceptronWeights { get; set; }
        public OutputType OutputLayerFlag { get; set; }
        public List<double> LittleError { get; set; }
        public List<double> Outputs { get; set; }

        public List<double> GetErrorVector(List<double> desiredOutputs, bool setVector = true)
        {
            var error = new List<double>();

            // desired outputs will always be 1-1 with the output layer perceptrons
            Enumerable.Range(0, desiredOutputs.Count).ToList()
                .ForEach(i => error.Add(Perceptrons.ElementAt(i).CalculateError(desiredOutputs.ElementAt(i))));

            if (setVector)
                LittleError = error;

            return error;
        }

        // every value is the given perceptron's activation function value
        public List<double> GetLayerOutputVector(List<double> input, bool setVector = true)
        {
            var layerOutput = new List<double>();

            foreach (var perceptron in Perceptrons)
            {
                perceptron.CalculateActivity(input);
                layerOutput.Add(perceptron.CalculateActivation(input));
            }

            if (setVector)
                Outputs = layerOutput;

            return layerOutput;
        }

        // input is the error vector
        // set delta vectors for this entire layer
        public void SetOutputLayerDeltas(List<double> errors)
        {
            Enumerable.Range(0, errors.Count).ToList()
                .ForEach(i => Perceptrons.ElementAt(i).SetDelta(errors.ElementAt(i)));
        }

        // input is above layer
        // sets delta values for entire layer
        // this provided layer must have the activity, activation value, and desired outputs set
        public void SetHiddenLayerDeltas(List<double> errors, Layer previousLayer)
        {
            Enumerable.Range(0, previousLayer.Perceptrons.Count).ToList()
                .ForEach(i => previousLayer.Perceptrons.ElementAt(i).SetDelta(errors.First()));
        }

        // calculates the delta weight for each perceptron in the provided layer
        // input argument is for the entire layer (provided)
        public void CalculateLayerDeltaWeights(List<List<double>> inputs, Layer someLayer)
        {
            if (inputs.Count == someLayer.Perceptrons.Count)
                Enumerable.Range(0, someLayer.Perceptrons.Count).ToList()
                .ForEach(i => someLayer.Perceptrons.ElementAt(i).SetDeltaWeights(inputs.ElementAt(i), someLayer.Perceptrons.ElementAt(i).Delta));
            else
                Enumerable.Range(0, someLayer.Perceptrons.Count).ToList()
                .ForEach(i => someLayer.Perceptrons.ElementAt(i).SetDeltaWeights(inputs.First(), someLayer.Perceptrons.ElementAt(i).Delta));
        }

        // updates the weights of the provided layer's perceptrons
        public void UpdateLayerWeights(Layer someLayer)
        {
            foreach(var perceptron in someLayer.Perceptrons)
            {
                perceptron.UpdateWeights();
            }
        }
    }
}
