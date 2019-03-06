using NeuralNetworkHelper.NeuralNetworkHelper.Domain;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace NeuralNetworkHelper.NeuralNetworkHelper.Logic
{
    public class PerceptronDeltaCalulator
    {
        public void RunIterations(Perceptron<List<double>> perceptron, List<double> input, double desiredOutput, double eta, int iterations)
        {
            var range = Enumerable.Range(1, iterations).ToList();
            foreach (var iteration in range)
            {
                perceptron.CalculateActivity(input);
                perceptron.CalculateActivation(input);
                var error = perceptron.CalculateError(desiredOutput, true);

                // we need to do this before we update the weights
                Debug.WriteLine($"Output:{Environment.NewLine}{Environment.NewLine}" +
                    $"Iteration:{iteration}{Environment.NewLine}" +
                    $"Input:{string.Join(",", input.ToArray())}{Environment.NewLine}" +
                    $"Weights:{string.Join(",", perceptron.Weights.ToArray())}{Environment.NewLine}" +
                    $"Activity:{perceptron.Activity}{Environment.NewLine}" +
                    $"Activation:{perceptron.Activation}{Environment.NewLine}" +
                    $"Error:{error}{Environment.NewLine}" +
                    $"Delta:{perceptron.Delta}{Environment.NewLine}{Environment.NewLine}");
                
                perceptron.SetDeltaWeights(input, eta);
                perceptron.UpdateWeights();

            }
            
        }
    }
}
