import random
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import gym
from fn_framework import FNAgent, Trainer, Observer


class ValueFunctionAgent(FNAgent):

    def save(self, model_path):  # モデル（価値関数）の保存
        joblib.dump(self.model, model_path)  # scikit-learnで学習したモデルを保存

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):  # モデル（価値関数）のロード
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path)  # scikit-learnで学習したモデルをロード
        agent.initialized = True
        return agent

    def initialize(self, experiences):  # モデル（価値関数）の初期化（モデル構築、正規化）
        scaler = StandardScaler()  # 後のscalar.fit(x)でxの標準化
        estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1)
            # ノード数10の隠れ層 x 2
            # max_iter: 最大何回学習を行うか
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])
        # 参考：scikit-learnを用いた機械学習パイプラインの作成 https://datadriven-rnd.com/ml-pipeline/
            # Pipeline=複数の一連の処理をまとめる（例：前処理＋学習・予測）。ここでは、
            #   scalar : 正規化処理のための変換器
            #   estimator : 価値関数本体

        # 正規化処理
        states = np.vstack([e.s for e in experiences])  # experiencesのsの履歴を結合したもの
            # np.vstack: ndarrayを垂直方向に結合
            # e = Experience(s, a, reward, n_state, done)
        self.model.named_steps["scaler"].fit(states)  # 正規化処理s
            # named_stepsメソッド: pipelineの各処理の名前をkeyとして各処理のオブジェクト？を取り出す

        # Avoid the predict before fit.
            # 学習する前に予測を行うと例外が発生するというscikit-learnの仕様を回避。１件の経験だけで一旦経験（update）を行っておく
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        print("Done initialization. From now, begin training!")


    def estimate(self, s):  # モデル（価値関数）を使った価値予測（行動選択時、policy内で使用、現在sのみから）
        estimated = self.model.predict(s)[0]  # [0]がついているのはなぜ？
        return estimated

    def _predict(self, states):  # モデル（価値関数）を使った価値予測（学習時、updateの中で使用、経験サンプルから）
        # initialized=Trueなら予測、それ以外は一様分布（初期値？）
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds


    # 学習（パラメータの更新）
        # train_loop内のstepで使用
        # 入力experiencesは、既にbatch_size分サンプリングされたbatch（Experience Replay）
    def update(self, experiences, gamma):
        states = np.vstack([e.s for e in experiences])  # experiencesに格納された全てのeについてのsの履歴
        n_states = np.vstack([e.n_s for e in experiences])  # experiencesに格納された全てのeについてのn_statesの履歴

        estimateds = self._predict(states)  # statesからの価値の予測
        future = self._predict(n_states)  # n_statesからの価値の予測

        for i, e in enumerate(experiences):  # estimatedの更新
            reward = e.r  # 実際の行動から得られた報酬
            if not e.d:
                reward += gamma * np.max(future[i])  # 実際の行動からの遷移先の最大価値
            estimateds[i][e.a] = reward
                # estimatedsの更新
                # （予測結果estimatedsのうち、実際に取った行動の部分を「得られた報酬」＋「遷移先の価値」で更新）

        estimateds = np.array(estimateds)
        states = self.model.named_steps["scaler"].transform(states)  # 正規化処理
        self.model.named_steps["estimator"].partial_fit(states, estimateds)
            # TD誤差（statesからのestimatedsと更新後のestimatedsの平均二乗誤差）が最小となるようにパラメータ調整
            # 入力：states, 出力：estimateds
            # partial_fit: 与えたデータに対して1エポックの学習。fitと違い、classesを引数に与える必要。
            #               fitだと、これまでの学習結果をリセットしてゼロから学習してしまう
            # named_steps: pipelineの各処理の名前をkeyとして各処理のオブジェクト？を取り出す


class CartPoleObserver(Observer):  # 環境envのラッパー。transformで、envから得られる状態をエージェントが扱いやすい形に変換

    def transform(self, state):  # envから得られる状態をエージェントが扱いやすい形に変換
        return np.array(state).reshape((1, -1))
            # 1次元配列に対してreshape(1, -1)とすると、その配列を要素とする2次元1行の配列に
            # 1次元配列に対してreshape(-1, 1)とすると、その配列を要素とする2次元1列の配列とする縦ベクトル


class ValueFunctionTrainer(Trainer):

    # train_loopを実施し、学習（パラメータ更新）したagentを返す
    def train(self, env, episode_count=220, epsilon=0.1, initial_count=-1,
              render=False):
        actions = list(range(env.action_space.n))
        agent = ValueFunctionAgent(epsilon, actions)
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent  # train_loop内のstepで学習（update）したエージェント

    def begin_train(self, episode, agent):  # initialize（モデルの作成、experiencesで正規化処理）の実施
        agent.initialize(self.experiences)

    # experiencesからbatch_size分のbatchをサンプリング、update（学習、Agent内で定義）を実行（train_loopの中で）
    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)  # agentの学習（パラメータ更新、Agent内で定義）

    def episode_end(self, episode, step_count, agent):  # 報酬合計の記録、
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play):
    env = CartPoleObserver(gym.make("CartPole-v0"))
    trainer = ValueFunctionTrainer()
    path = trainer.logger.path_of("value_function_agent.pkl")

    if play:   # episode_count分シミュレーション実施（学習済みのモデルの動作確認のため）
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
