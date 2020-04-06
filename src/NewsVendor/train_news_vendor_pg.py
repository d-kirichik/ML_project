class MyLauncher(SageMakerRayLauncher):

    def register_env_creator(self):
        from news_vendor_environment import NewsVendorGymEnvironmentNormalized
        register_env("NewsVendorGymEnvironment-v1", lambda env_config: NewsVendorGymEnvironmentNormalized(env_config))

    def get_experiment_config(self):
        return {
            "training": {
                "env": "NewsVendorGymEnvironment-v1",
                "run": "PG",
                "config": {
                    "ignore_worker_failures": True,
                    "train_batch_size": 320000,
                    "num_workers": (self.num_cpus - 1),
                    "num_gpus": self.num_gpus,
                    "batch_mode": "complete_episodes",
                    "env_config": {
                    },
                    'observation_filter': 'MeanStdFilter',
                }
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()
