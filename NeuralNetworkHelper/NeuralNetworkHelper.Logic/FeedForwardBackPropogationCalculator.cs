using NeuralNetworkHelper.NeuralNetworkHelper.Domain;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace NeuralNetworkHelper.NeuralNetworkHelper.Logic
{
    public class FeedForwardBackPropogationCalculator
    {
        public void RunEpochs(List<double> input, Layer hiddenLayer, Layer outputLayer, double desiredOutput, double eta, int epochs)
        {
            // i'm going to assume there are only 2 layers and inputs,
            // 2 inputs, one hidden layer of two nodes, 1 output node from that hidden layer
            
            var range = Enumerable.Range(1, epochs).ToList();
            foreach (var iteration in range)
            {
                // calculate activity and activation of hidden layer
                hiddenLayer.Perceptrons
                    .ForEach(p =>
                        {
                            p.CalculateActivity(input);
                            p.CalculateActivation(input);
                        }
                    );

                // get activations from hidden layer to act as inputs for output layer
                var finalNodeInput = hiddenLayer.Perceptrons.Select(p => p.Activation).ToList();

                // calculate activity, activation, error, delta, delta weights for output layer
                outputLayer.Perceptrons
                    .ForEach(p =>
                        {
                            p.CalculateActivity(finalNodeInput);
                            p.CalculateActivation(finalNodeInput);
                            p.SetDelta(p.CalculateError(desiredOutput));
                            p.SetDeltaWeights(p.CurrentInput, eta);
                        }
                    );

                // get and set the error vector for output layer
                var errors = outputLayer.GetErrorVector(new List<double> { desiredOutput });

                // grab the output node delta
                var outputDelta = outputLayer.Perceptrons.First().Delta;

                // calculate the hidden layer updated perceptron deltas
                Enumerable.Range(0, hiddenLayer.Perceptrons.Count).ToList()
                    .ForEach(i =>
                        hiddenLayer.Perceptrons.ElementAt(i).Delta
                            = (1 - hiddenLayer.Perceptrons.ElementAt(i).Activation)
                            *(hiddenLayer.Perceptrons.ElementAt(i).Activation)
                            *(outputDelta)
                            *(outputLayer.Perceptrons.First().Weights.ElementAt(i)));

                // calculate the updated weights for the input to hidden layer
                hiddenLayer.Perceptrons
                    .ForEach(p =>
                    {
                        // weight_i + eta*OutputDelta*input_i
                        Enumerable.Range(0, p.Weights.Count).ToList()
                            .ForEach(i =>
                            {
                                p.Weights[i] = (p.Weights.ElementAt(i))+(p.Delta)*(eta)*(p.CurrentInput.ElementAt(i));
                            });
                    });

                outputLayer.Perceptrons.ForEach(p => p.UpdateWeights());

                PrintInfoPerEpoch(input, hiddenLayer, outputLayer, desiredOutput, eta, iteration);
            }
        }

        public void PrintInfoPerEpoch(List<double> input, Layer hiddenLayer, Layer outputLayer, double desiredOutput, double eta, int epoch)
        {
            Debug.WriteLine($"Epoch-{epoch} Output:{Environment.NewLine}{Environment.NewLine}");
            Debug.WriteLine($"Input:{string.Join(",", input.ToArray())}{Environment.NewLine}");
            Debug.WriteLine($"{Environment.NewLine}Hidden Layer Perceptrons:{Environment.NewLine}");
            foreach (var perceptron in hiddenLayer.Perceptrons)
            {
                WriteInforForPerceptron(perceptron);
            }
            Debug.WriteLine($"{Environment.NewLine}Output Layer Perceptrons:{Environment.NewLine}");
            foreach (var perceptron in outputLayer.Perceptrons)
            {
                WriteInforForPerceptron(perceptron);
            }
        }

        public void WriteInforForPerceptron(Perceptron<List<double>> perceptron)
        {
            Debug.WriteLine($"Weights:{string.Join(",", perceptron.Weights.ToArray())}{Environment.NewLine}" +
                    $"Activity:{perceptron.Activity}{Environment.NewLine}" +
                    $"Activation:{perceptron.Activation}{Environment.NewLine}" +
                    $"Delta:{perceptron.Delta}{Environment.NewLine}{Environment.NewLine}");
        }
    }
}
