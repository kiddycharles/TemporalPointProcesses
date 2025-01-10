================================
Evaluate a Model
================================

Step 1: Setup the config file
===============================================

Same as in the training pipeline, firstly we need to initialize the task configuration in the config file.

Similar to the setup in `Training Pipeline <./run_train_pipeline.html>`_, we set the `stage` to `eval` and pass the `pretrained_model_dir` to ``the model_config``

Note that the *pretrained_model_dir* can be found in the log of the training process.

.. code-block:: yaml

    NHP_eval:
      base_config:
        stage: eval
        backend: torch
        dataset_id: taxi
        runner_id: std_tpp
        base_dir: './checkpoints/'
        model_id: NHP
      trainer_config:
        batch_size: 256
        max_epoch: 1
      model_config:
        hidden_size: 64
        use_ln: False
        seed: 2019
        gpu: 0
        pretrained_model_dir: ./checkpoints/26507_4380788096_231111-101848/models/saved_model  # must provide this dir
        thinning:
          num_seq: 10
          num_sample: 1
          num_exp: 500 # number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
          look_ahead_time: 10
          patience_counter: 5 # the maximum iteration used in adaptive thinning
          over_sample_rate: 5
          num_samples_boundary: 5
          dtime_max: 5




A complete example of these files can be seen at `examples/example_config.yaml <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/configs/experiment_config.yaml>`_ .


Step 2: Run the evaluation script
=================================

Same as in the training pipeline, we need to initialize a ``ModelRunner`` object to do the evaluation.

The following code is an example, which is a copy from `examples/train_nhp.py <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/train_nhp.py>`_ .


.. code-block:: python

    import argparse

    from easy_tpp.config_factory import RunnerConfig
    from easy_tpp.runner import Runner


    def main():
        parser = argparse.ArgumentParser()

        parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                            help='Dir of configuration yaml to train and evaluate the model.')

        parser.add_argument('--experiment_id', type=str, required=False, default='RMTPP_eval',
                            help='Experiment id in the config file.')

        args = parser.parse_args()

        config = RunnerConfig.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

        model_runner = Runner.build_from_config(config)

        model_runner.run()


    if __name__ == '__main__':
        main()




Checkout the output
====================

The evaluation result will be print in the console and saved in the logs whose directory is specified in the
out config file, i.e.:

.. code-block:: bash

    'output_config_dir': './checkpoints/NHP_test_conttime_20221002-13:19:23/NHP_test_output.yaml'
