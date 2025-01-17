import random
import argparse
from collections import deque
import numpy as np
from tensorflow.python import keras as K
from PIL import Image
import gym
import gym_ple
from fn_framework import FNAgent, Trainer, Observer


class DeepQNetworkAgent(FNAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._scaler = None
        self._teacher_model = None

    def initialize(self, experiences, optimizer):
        #
        # optimizerはDeepQNetworkTrainerのbegin_trainからわたされる
        feature_shape = experiences[0].s.shape  # make_modelのinput_shapeに渡される
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()  # 活性化関数の初期化パラメータGlorot
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(256, kernel_initializer=normal,
                                 activation="relu"))
        model.add(K.layers.Dense(len(self.actions),
                                 kernel_initializer=normal))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)
            # 学習を安定化させるため、遷移先の価値を、モデル本体ではなく、一定期間固定されたteacherモデルのパラメータから
            # 算出するFixed Target Q-Networkの手法
            # .clone_model: モデルのクローン作成（ウェイトは継承されない）

    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]

    def update(self, experiences, gamma):  # 引数experiencesには、self.experiencesからサンプルしたバッジ
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])

        estimateds = self.model.predict(states)  # 現モデルによる、sに対するactionsそれぞれの価値
        future = self._teacher_model.predict(n_states)  # teacher_modelによる、n_sに対するactionsそれぞれの価値

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])  # teacher_modelでn_sに最大価値を与えるaction
            estimateds[i][e.a] = reward  # estimatedsのうち選択された行動の部分だけ価値を書き換える

        loss = self.model.train_on_batch(states, estimateds)
            # １つのバッジだけで勾配を更新し、訓練損失を返す、
        return loss

    def update_teacher(self):
        self._teacher_model.set_weights(self.model.get_weights())  # 本体モデルのウェイトで更新


class DeepQNetworkAgentTest(DeepQNetworkAgent):
    # CNNの学習は時間がかかるため、ネットワーク構造（make_model）以外の箇所の挙動を事前にテストするためのAgent

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Dense(64, input_shape=feature_shape,
                                 kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal,
                                 activation="relu"))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)


class CatcherObserver(Observer):
        # 時系列に並んだ４つの画面フレームをまとめる処理

    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # Convert the feature shape (f, w, h) => (h, w, f).
        feature = np.transpose(feature, (1, 2, 0))

        return feature


class DeepQNetworkTrainer(Trainer):

    def __init__(self, buffer_size=50000, batch_size=32,
                 gamma=0.99, initial_epsilon=0.5, final_epsilon=1e-3,
                 learning_rate=1e-3, teacher_update_freq=3, report_interval=10,
                 log_dir="", file_name=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self.file_name = file_name if file_name else "dqn_agent.h5"
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.teacher_update_freq = teacher_update_freq  # 訓練開始後からのepisode数がteacher_update_freqごとにupdate_teacher
        self.loss = 0
        self.training_episode = 0  # epsilonのdecayの制御などに使用
        self._max_reward = -10

    def train(self, env, episode_count=1200, initial_count=200,
              test_mode=False, render=False, observe_interval=100):
        actions = list(range(env.action_space.n))

        # test_modeの制御
        if not test_mode:
            agent = DeepQNetworkAgent(1.0, actions)
        else:
            agent = DeepQNetworkAgentTest(1.0, actions)
            observe_interval = 0

        self.training_episode = episode_count  # epsilonのdecayに使用

        # episode_count分だけ学習ループを実施
        self.train_loop(env, agent, episode_count, initial_count, render,
                        observe_interval)
        return agent

    # エピソード開始（損失の初期化）
    def episode_begin(self, episode, agent):
        self.loss = 0

    # 訓練開始（optimizerの設定、agentの初期化、モデルの設定など）
    def begin_train(self, episode, agent):
        optimizer = K.optimizers.Adam(lr=self.learning_rate, clipvalue=1.0)  # Optimizer=Adam
        agent.initialize(self.experiences, optimizer)
        self.logger.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode  # 既に進んだエピソード分だけ、training_episode(=episode_count=1200)から引く？

    # 学習（訓練開始後なら、batchをサンプルして学習）
    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            self.loss += agent.update(batch, self.gamma)  # 訓練損失に追加？

    # エピソード終了
    def episode_end(self, episode, step_count, agent):
            # step_count: 各episodeがdoneまでにかかったstep数

        # 報酬計の計算・記録
        reward = sum([e.r for e in self.get_recent(step_count)])
        self.loss = self.loss / step_count
        self.reward_log.append(reward)

        # 訓練開始後の場合
        if self.training:

            # サマリの表示
            self.logger.write(self.training_count, "loss", self.loss)
            self.logger.write(self.training_count, "reward", reward)
            self.logger.write(self.training_count, "epsilon", agent.epsilon)

            # 報酬計が過去最高の場合 ⇒ 過去最高の更新
            if reward > self._max_reward:
                agent.save(self.logger.path_of(self.file_name))
                self._max_reward = reward

            # 訓練開始後からのepisode数がteacher_update_freqごとにupdate_teacher
            if self.is_event(self.training_count, self.teacher_update_freq):
                agent.update_teacher()

            # epsilonのdecay
            diff = (self.initial_epsilon - self.final_epsilon)
            decay = diff / self.training_episode
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)  # 訓練進行に合わせてepsilonを減衰

        # 直近report_interval分（episode数）の、各step、各episodeの報酬計の平均・分散の表示
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
                # 各episodeの報酬計の履歴のうち直近report_interval分を取り出す
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play, is_test):
    file_name = "dqn_agent.h5" if not is_test else "dqn_agent_test.h5"
    trainer = DeepQNetworkTrainer(file_name=file_name)
    path = trainer.logger.path_of(trainer.file_name)
    agent_class = DeepQNetworkAgent

    if is_test:
        print("Train on test mode")
        obs = gym.make("CartPole-v0")
        agent_class = DeepQNetworkAgentTest
    else:
        env = gym.make("Catcher-v0")
        obs = CatcherObserver(env, 80, 80, 4)
        trainer.learning_rate = 1e-4

    if play:
        agent = agent_class.load(obs, path)
        agent.play(obs, render=True)
    else:
        trainer.train(obs, test_mode=is_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")
    parser.add_argument("--test", action="store_true",
                        help="train by test mode")

    args = parser.parse_args()
    main(args.play, args.test)
