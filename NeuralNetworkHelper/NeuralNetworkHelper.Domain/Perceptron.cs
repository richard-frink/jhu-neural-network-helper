using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkHelper.NeuralNetworkHelper.Domain
{
    // T is the input vector (datatype will be used to store weights and input)
    // this might change over time, but it probably always going to be a list with some datatype
    public class Perceptron<T>
        where T : List<double>
    {
        // this is what we'll use for a lot of cases but this will change later on
        public Perceptron()
        {
            ActivtyFunction = new Func<Perceptron<T>, double>
            (
                p =>
                    Enumerable.Range(0, p.Weights.Count)
                    .Sum(i => p.Weights.ElementAt(i)*p.CurrentInput.ElementAt(i))
                    + p.Bias
            );

            ActivationFunction = new Func<Perceptron<T>, double>
            (
                p =>
                    1/(1+Math.Exp((-1)*p.Activity))
            );
        }

        public T Weights { get; set; }
        public T DeltaWeights { get; set; }

        public T CurrentInput { get; set; }

        public double Bias { get; set; }
        public double DeltaBias { get; set; }

        public double Activity { get; set; }
        public double Activation { get; set; }

        public double Delta { get; set; }

        // activity function as result float
        public Func<Perceptron<T>, double> ActivtyFunction { get; set; }

        // activation function as result float
        public Func<Perceptron<T>, double> ActivationFunction { get; set; }

        public double CalculateActivity(T input, bool updateMyActivity = true)
        {
            var activity = ActivtyFunction.Invoke(this);

            if (updateMyActivity)
                Activity = activity;

            return activity;
        }

        public double CalculateActivation(T input, bool updateMyActivation = true)
        {
            var activation = ActivationFunction.Invoke(this);

            if (updateMyActivation)
                Activation = activation;

            return activation;
        }

        public void SetDeltaWeights(T input, double eta)
        {
            DeltaWeights.Clear();

            Enumerable.Range(0, input.Count).ToList().ForEach(i => DeltaWeights.Add(input.ElementAt(i)*eta*Delta));
        }

        public void UpdateWeights()
        {
            var tempWeights = Weights.ToArray();
            Enumerable.Range(0, DeltaWeights.Count).ToList().ForEach(i => tempWeights[i] = tempWeights[i] + DeltaWeights.ElementAt(i));

            Weights.Clear();
            Enumerable.Range(0, tempWeights.Length).ToList().ForEach(i => Weights.Add(tempWeights[i]));
        }

        // it may be useful to store the current input, maybe...
        public void UpdateInput(T input)
        {
            CurrentInput.Clear();
            input.ForEach(e => CurrentInput.Add(e));
        }

        public double CalculateError(double desiredOutput, bool setDelta = false)
        {
            var error = desiredOutput - Activation;

            if (setDelta)
                SetDelta(error);

            return error;
        }

        public void SetDelta(double error)
        {
            Delta = Activation * (1 - Activation) * error;
        }
    }
}
