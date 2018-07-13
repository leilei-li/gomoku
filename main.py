from env.play_a_game import run
from agent.train import TrainPipeline

if __name__ == '__main__':
    training_pipeline = TrainPipeline(8, 8, 5)
    training_pipeline.run()
