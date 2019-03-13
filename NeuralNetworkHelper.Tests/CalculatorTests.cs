using NeuralNetworkHelper.NeuralNetworkHelper.Domain;
using NeuralNetworkHelper.NeuralNetworkHelper.Logic;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Tests
{
    public class CalculatorTests
    {
        private PerceptronDeltaCalulator _iterationCalulator;
        private FeedForwardBackPropogationCalculator _epochCalculator;

        [SetUp]
        public void Setup()
        {
            _iterationCalulator = new PerceptronDeltaCalulator();
            _epochCalculator = new FeedForwardBackPropogationCalculator();
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

            _iterationCalulator.RunIterations(perceptron, input, desiredOutput, eta, 3);

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

            _iterationCalulator.RunIterations(perceptron, input, desiredOutput, eta, 31);

            Debug.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }

        [Test]
        public void RunEpochs_ModuleFiveExample()
        {
            var input = new List<double> { 1, 2 };
            var hiddenLayerWeights1 = new List<double> { .3, .3 };
            var hiddenLayerWeights2 = new List<double> { .3, .3 };
            var outputLayerWeights = new List<double> { .8, .8 };
            var bias = 0;
            var desiredOutput = .7;
            var eta = 1;

            // init layers
            Layer hiddenLayer =
                new Layer(
                    OutputType.HiddenLayer,
                    new List<Perceptron<List<double>>>
                    {
                        new Perceptron<List<double>>
                        {
                            Bias = bias,
                            Weights = hiddenLayerWeights1,
                            DeltaWeights = new List<double>(),
                            CurrentInput = input
                        },
                        new Perceptron<List<double>>
                        {
                            Bias = bias,
                            Weights = hiddenLayerWeights2,
                            DeltaWeights = new List<double>(),
                            CurrentInput = input
                        }
                    }
                );
            Layer outputLayer =
                new Layer(
                    OutputType.HiddenLayer,
                    new List<Perceptron<List<double>>>
                    {
                        new Perceptron<List<double>>
                        {
                            Bias = bias,
                            Weights = outputLayerWeights,
                            DeltaWeights = new List<double>(),
                            CurrentInput = input
                        }
                    }
                );

            _epochCalculator.RunEpochs(input, hiddenLayer, outputLayer, desiredOutput, eta, 3);

            Debug.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }

        [Test]
        public void RunEpochs_ModuleSixProblems()
        {
            var input = new List<double> { 1, 3 };
            var hiddenLayerWeights1 = new List<double> { .8, .1 };
            var hiddenLayerWeights2 = new List<double> { .5, .2 };
            var outputLayerWeights = new List<double> { .2, .7 };
            var bias = 0;
            var desiredOutput = .95;
            var eta = .1;

            // init layers
            Layer hiddenLayer =
                new Layer(
                    OutputType.HiddenLayer,
                    new List<Perceptron<List<double>>>
                    {
                        new Perceptron<List<double>>
                        {
                            Bias = bias,
                            Weights = hiddenLayerWeights1,
                            DeltaWeights = new List<double>(),
                            CurrentInput = input
                        },
                        new Perceptron<List<double>>
                        {
                            Bias = bias,
                            Weights = hiddenLayerWeights2,
                            DeltaWeights = new List<double>(),
                            CurrentInput = input
                        }
                    }
                );
            Layer outputLayer =
                new Layer(
                    OutputType.HiddenLayer,
                    new List<Perceptron<List<double>>>
                    {
                        new Perceptron<List<double>>
                        {
                            Bias = bias,
                            Weights = outputLayerWeights,
                            DeltaWeights = new List<double>(),
                            CurrentInput = input
                        }
                    }
                );

            _epochCalculator.RunEpochs(input, hiddenLayer, outputLayer, desiredOutput, eta, 3);

            Debug.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }
    }
}