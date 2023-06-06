import os
import random
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.python import keras as K
import gym
from fn_framework import FNAgent, Trainer, Observer, Experience
tf.compat.v1.disable_eager_execution()


class PolicyGradientAgent(FNAgent):

    def __init__(self, actions):
        # PolicyGradientAgent uses self policy (doesn't use epsilon).
        super().__init__(epsilon=0.0, actions=actions)  # epsilon=0、すべての行動は戦略に基づき決定
        self.estimate_probs = True  # 予測対象は行動確率。FNAgentのpolicyで使う
        self.scaler = StandardScaler()  # 後のscalar.fit(x)でxの標準化
        self._updater = None  # 後のset_updaterで定義

    def save(self, model_path):
        super().save(model_path)
        joblib.dump(self.scaler, self.scaler_path(model_path))

    @classmethod
    def load(cls, env, model_path):  # 環境とパスを入力として、agentとモデルをロード（playで使う）
        actions = list(range(env.action_space.n))
        agent = cls(actions)  # __init__(self, actions)が参照されるので、agent = PolicyGradientAgent(actions)と同じ効果？
        agent.model = K.models.load_model(model_path)  # モデルのロード
        agent.initialized = True
        agent.scaler = joblib.load(agent.scaler_path(model_path))
        return agent

    def scaler_path(self, model_path):
        fname, _ = os.path.splitext(model_path)
        fname += "_scaler.pkl"
        return fname

    def initialize(self, experiences, optimizer):  # モデル（価値関数）の初期化（モデル構築、正規化）
        states = np.vstack([e.s for e in experiences])
        feature_size = states.shape[1]
        self.model = K.models.Sequential([
            K.layers.Dense(10, activation="relu", input_shape=(feature_size,)),
            K.layers.Dense(10, activation="relu"),
            K.layers.Dense(len(self.actions), activation="softmax")
            # 入力：states
            # 出力：今回は、出力は各状態の行動の価値ではなく、各行動の確率。そのため活性化関数はsoftmax
        ])
        self.set_updater(optimizer)  # オプティマイザーにAdamsを指定して方策パラメータ更新
        self.scaler.fit(states)  # statesの正規化
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def set_updater(self, optimizer):  # 方策パラメータの更新
        actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
        rewards = tf.compat.v1.placeholder(shape=(None), dtype="float32")
            # プレースホルダーの設定
            # 注意：compat.v1.placeholderはTF2のeager executionとcompatibleでない
            # To migrate to TF2, rewrite the code to be compatible with eager execution.
            # https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1)
            # tf.one_hot: indicesで示した部分をon_value、それ以外をoff_value
            # 今回の場合、len(self.actions)のサイズのセルのうち、actionsを1, それ以外を0（図4-28）
            # でも、actionsはlist(range(env.action_space.n))ではないのか？理解できない
            # https://j138i.com/tf-one-hot
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs,
                                              axis=1)  # 各行動の確率pi(a|s)
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)  # log 0 を避けるため、確率地の範囲を調整
        loss = - tf.math.log(clipped) * rewards  # log pi(a|s) Q(s,a)
        loss = tf.reduce_mean(loss)  # E[log pi(a|s) Q(s,a)]

        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)
            # 勾配計算の完了: Delta E[log pi(a|s) Q(s,a)]
            # get_updates:  https://github.com/keras-team/keras/blob/v2.12.0/keras/optimizers/legacy/optimizer_v2.py#L868-L878

        self._updater = K.backend.function(
                                        inputs=[self.model.input,
                                                actions, rewards],
                                        outputs=[loss],
                                        updates=updates)
            # 状態（self.model.input）と実際の行動（actions）、その結果の価値（rewards）を引数として、
            # パラメータの更新（updates）を行う関数を定義（後のupdateの中で使う）
            # kares.backend：イマイチよくわからない。おそらく、モジュール内のこれまでの記述を関数にまとめる役割

    def estimate(self, s):
        normalized = self.scaler.transform(s)
        action_probs = self.model.predict(normalized)[0]
        return action_probs

    def update(self, states, actions, rewards):
        normalizeds = self.scaler.transform(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        self._updater([normalizeds, actions, rewards])
            # [状態, 実際の行動（actions）, その結果の価値（rewards）]を引数としてパラメータの更新を行う関数（set_updaterで定義）


class CartPoleObserver(Observer):

    def transform(self, state):
        return np.array(state).reshape((1, -1))


class PolicyGradientTrainer(Trainer):

    def __init__(self, buffer_size=256, batch_size=32, gamma=0.9,
                 report_interval=10, log_dir=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)

    def train(self, env, episode_count=220, initial_count=-1, render=False):
        actions = list(range(env.action_space.n))
        agent = PolicyGradientAgent(actions)
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def episode_begin(self, episode, agent):
        if agent.initialized:
            self.experiences = []

    def make_batch(self, policy_experiences):
        length = min(self.batch_size, len(policy_experiences))
        batch = random.sample(policy_experiences, length)
        states = np.vstack([e.s for e in batch])
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        scaler = StandardScaler()
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = scaler.fit_transform(rewards).flatten()
        return states, actions, rewards

    def episode_end(self, episode, step_count, agent):
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        if not agent.initialized:
            if len(self.experiences) == self.buffer_size:
                optimizer = K.optimizers.Adam(lr=0.01)
                agent.initialize(self.experiences, optimizer)
                self.training = True
        else:
            policy_experiences = []
            for t, e in enumerate(self.experiences):
                s, a, r, n_s, d = e
                d_r = [_r * (self.gamma ** i) for i, _r in
                       enumerate(rewards[t:])]
                d_r = sum(d_r)
                d_e = Experience(s, a, d_r, n_s, d)
                policy_experiences.append(d_e)

            agent.update(*self.make_batch(policy_experiences))

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play):
    env = CartPoleObserver(gym.make("CartPole-v0"))
    trainer = PolicyGradientTrainer()
    path = trainer.logger.path_of("policy_gradient_agent.h5")

    if play:
        agent = PolicyGradientAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
