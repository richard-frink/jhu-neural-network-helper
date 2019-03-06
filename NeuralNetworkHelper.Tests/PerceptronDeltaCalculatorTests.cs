using NeuralNetworkHelper.NeuralNetworkHelper.Domain;
using NeuralNetworkHelper.NeuralNetworkHelper.Logic;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Tests
{
    public class PerceptronDeltaCalculatorTests
    {
        private PerceptronDeltaCalulator _calulator; 

        [SetUp]
        public void Setup()
        {
            _calulator = new PerceptronDeltaCalulator();
        }

        [Test]
        public void RunIterations_ModuleFourPerceptron()
        {
            var input = new List<double> { 1, 0 };
            var startingWeights = new List<double> { -.3, .6 };
            var bias = .2;
            var desiredOutput = .8;
            var eta = .1;

            // init perceptron
            Perceptron<List<double>> perceptron = new Perceptron<List<double>>
            {
                Bias = bias,
                Weights = startingWeights,
                DeltaWeights = new List<double>(),
                CurrentInput = input
            };

            _calulator.RunIterations(perceptron, input, desiredOutput, eta, 3);

            Debug.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }

        [Test]
        public void RunIterations_ModuleFiveQOne()
        {
            var input = new List<double> { .8, .9 };
            var startingWeights = new List<double> { .24, .88 };
            var bias = 0;
            var desiredOutput = .15;
            var eta = 5;

            // init perceptron
            Perceptron<List<double>> perceptron = new Perceptron<List<double>>
            {
                Bias = bias,
                Weights = startingWeights,
                DeltaWeights = new List<double>(),
                CurrentInput = input
            };

            _calulator.RunIterations(perceptron, input, desiredOutput, eta, 31);

            Debug.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }
    }
}