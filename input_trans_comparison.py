from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from sys import argv, stdout, stderr
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experimenter import Experimenter
from random import sample
from string import ascii_uppercase


def run_input_trans_comparison(
        n,
        k,
        transformations,
        Ns,
        combiner=LTFArray.combiner_xor,
        instance_sample_size=100,
        seed_instance=0x15eed,
        initial_seed_model=0x5eed,
        log_file_prefix = None
    ):
    """
    This function runs experiments to compare different input transformations regarding their
    influence on the learnability by Logistic Regression.
    For a given size of LTFArray (n,k), and a given list of input transformations, and a given
    list of number of CRPs in the training set (Ns), for each combination of input transformation
    and training set size, an LTF Array instance is created. Each LTF Array will be learned using
    instance_sample_size attempts with different initializations.
    All LTF Arrays are using the given combiner function. All results are written the log files
    with the given prefix.
    """

    # Set up logging
    if log_file_prefix is None:
        log_file_prefix = ''.join(sample(list(ascii_uppercase), 5))
    stderr.write('log file prefix: %s\n' % log_file_prefix)
    stderr.write('running %s experiments' % str(len(transformations)*len(Ns)*instance_sample_size))

    # Experimenter instance that is used for experiment scheduling
    e = Experimenter(log_file_prefix, [])

    # Add all transformation/N-combinations to the Experimenter Queue
    for transformation in transformations:
        for N in Ns:
            seed_model = initial_seed_model
            for j in range(instance_sample_size):
                e.experiments.append(
                    ExperimentLogisticRegression(
                        log_name=log_file_prefix,
                        n=n,
                        k=k,
                        N=N,
                        seed_model=seed_model,
                        seed_instance=seed_instance,
                        transformation=transformation,
                        combiner=combiner,
                    )
                )
                seed_model += 1

    # Run all experiments
    e.run()

if __name__ == '__main__':
    run_input_trans_comparison(
        n=64,
        k=2,
        transformations=[
            LTFArray.transform_concat(
                transform_1=LTFArray.transform_shift,
                transform_2=LTFArray.transform_id,
                nn=48
            )
            #LTFArray.transform_1_1_bent,
            #LTFArray.transform_1_n_bent,
            #LTFArray.transform_atf,
            #LTFArray.transform_id,
            #LTFArray.transform_lightweight_secure,
            #LTFArray.transform_mm,
            #LTFArray.transform_polynomial,
            #LTFArray.transform_shift,
            #LTFArray.transform_shift_lightweight_secure,
            #LTFArray.transform_soelter_lightweight_secure,
        ],
        Ns=[
            250,
            1000,
            2000,
            #12000,
            #12000
            #30016,
            #100000,
            #250016,
            #1000000,
        ],
        instance_sample_size=10,
        initial_seed_model=0xdead,
    )
